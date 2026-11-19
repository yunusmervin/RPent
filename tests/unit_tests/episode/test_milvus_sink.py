# Copyright 2026 The IIE Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.libero.robot_spec import _embedding_requested
from rpent.cli.main import _build_argparser
from rpent.episode.hook import record_episode
from rpent.episode.milvus_sink import (
    VECTOR_DIM,
    MilvusEpisodeSink,
    episode_collection_name,
)
from rpent.episode.model import Episode, EpisodeAssets, EpisodeStats
from rpent.episode.writer import EpisodeWriter
from rpent.memory import AGFSBackend
from rpent.utils.config import EMBEDDING_MODEL_ID
from tools.rebuild_episode_index import LocalMockEmbedding
from tools.rebuild_episode_index import main as rebuild_main

CREATED = datetime(2026, 11, 10, 3, 15, tzinfo=UTC)
VERSION = "GPT_6_astra_low"


class HashEmbedding:
    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]:
        content = json.dumps([text, image_uris or []]).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(content).digest()[:8], "big")
        vector = np.random.default_rng(seed).random(VECTOR_DIM, dtype=np.float32)
        return {
            "vector": vector.tolist(),
            "model": EMBEDDING_MODEL_ID,
            "dim": VECTOR_DIM,
        }


def _backend(root: Path) -> AGFSBackend:
    return AGFSBackend(
        root / "memory" / "libero",
        agfs_endpoint="http://127.0.0.1:18080",
        robot="libero",
        record_episodes=True,
    )


def _episode(
    episode_id: str,
    *,
    robot: str = "libero",
    memory_version: str = VERSION,
    first_frame: str | None = None,
) -> Episode:
    return Episode(
        episode_id=episode_id,
        cell_tag="libero10_swap_t2_s0",
        robot=robot,
        suite="libero_10_swap",
        regime="swap",
        task_id=2,
        seed=0,
        task_language="pick up the black bowl",
        outcome="success",
        confidence="verified",
        memory_ids=["task-family_libero10_swap_t2"],
        memory_version=memory_version,
        embedding_model=EMBEDDING_MODEL_ID,
        created_at=CREATED,
        stats=EpisodeStats(elapsed_s=1, tool_calls=2, output_tokens=3),
        assets=EpisodeAssets(
            first_frame=first_frame
            or f"agfs://frames/sha256/00/00/{0:064x}.png"
        ),
    )


def _put_episode(backend: AGFSBackend, episode: Episode) -> str:
    return EpisodeWriter(backend).write(episode)


def _search(
    sink: MilvusEpisodeSink,
    episode: Episode,
    embedding: Any,
    *,
    robot: str = "libero",
) -> list[str]:
    vector = embedding.encode(
        text=episode.task_language,
        image_uris=[episode.assets.first_frame],
    )["vector"]
    results = sink.client.search(
        collection_name=episode_collection_name(episode.memory_version),
        data=[vector],
        filter=f'robot == "{robot}"',
        limit=5,
        output_fields=["episode_id", "robot", "episode_uri", "memory_ids"],
    )
    return [result["entity"]["episode_id"] for result in results[0]]


def test_schema_properties_and_upsert_are_idempotent(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    versioned = _episode("episode-one", memory_version="GPT 6:astra")
    uri = _put_episode(backend, versioned)
    sink = MilvusEpisodeSink(HashEmbedding(), memory_root=backend.root)
    try:
        sink.upsert(versioned, uri)
        sink.upsert(versioned, uri)
        collection = "episodes__GPT_6_astra"
        description = sink.client.describe_collection(collection)
        fields = {field["name"]: field for field in description["fields"]}

        assert episode_collection_name("GPT 6:astra/low") == "episodes__GPT_6_astra_low"
        assert list(fields) == [
            "episode_id",
            "vector",
            "robot",
            "suite",
            "regime",
            "task_id",
            "seed",
            "outcome",
            "confidence",
            "memory_ids",
            "episode_uri",
        ]
        assert fields["episode_id"]["is_primary"] is True
        assert fields["episode_id"]["params"]["max_length"] == 128
        assert fields["vector"]["params"]["dim"] == 1536
        assert fields["robot"]["is_partition_key"] is True
        assert fields["memory_ids"]["element_type"].name == "VARCHAR"
        assert fields["memory_ids"]["params"]["max_capacity"] == 64
        assert fields["episode_uri"]["params"]["max_length"] == 512
        assert description["properties"]["embedding_model"] == EMBEDDING_MODEL_ID
        vector_index = sink.client.describe_index(collection, "vector")
        assert vector_index["index_type"] == "HNSW"
        assert vector_index["metric_type"] == "COSINE"
        assert sink.client.get_collection_stats(collection)["row_count"] == 1
        assert (backend.root / ".milvus" / "episodes.db").exists()
    finally:
        sink.close()


def test_robot_partition_filter_excludes_other_robots(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    libero = _episode("libero-episode")
    franka = _episode("franka-episode", robot="franka")
    libero_uri = _put_episode(backend, libero)
    franka_uri = _put_episode(backend, franka)
    sink = MilvusEpisodeSink(HashEmbedding(), memory_root=backend.root)
    try:
        sink.upsert_many([(libero, libero_uri), (franka, franka_uri)])

        assert _search(sink, libero, HashEmbedding()) == ["libero-episode"]
        assert _search(sink, libero, HashEmbedding(), robot="franka") == [
            "franka-episode"
        ]
    finally:
        sink.close()


def _run_directory(root: Path) -> Path:
    run = root / "run"
    frame = run / "agentview.png" / "00.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"first-frame")
    (run / "states.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "step_idx": 0,
                        "artifacts": ["agentview.png"],
                        "terminated": True,
                        "truncated": False,
                        "extras": {"task_language": "pick up the black bowl"},
                    }
                ]
            }
        )
    )
    return run


def _record_kwargs(backend: AGFSBackend, run: Path) -> dict[str, Any]:
    return {
        "run_state_dir": run,
        "cell_tag": "libero10_swap_t2_s0",
        "robot": "libero",
        "task_desc": {"suite": "libero_10_swap", "task": 2, "seed": 0},
        "finish_result": {"status": "success"},
        "stats": {},
        "elapsed_s": 1,
        "memory_version": VERSION,
        "created_at": CREATED,
    }


def test_agfs_write_precedes_encoding_and_upsert(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    expected_path = (
        backend.episodes_dir
        / VERSION
        / "libero10_swap_t2_s0_20261110T031500.json"
    )

    class OrderCheckingEmbedding(HashEmbedding):
        def encode(
            self, text: str = "", image_uris: list[str] | None = None
        ) -> dict[str, Any]:
            assert expected_path.is_file()
            return super().encode(text, image_uris)

    uri = record_episode(
        backend,
        **_record_kwargs(backend, _run_directory(tmp_path)),
        embedding_client=OrderCheckingEmbedding(),
    )

    assert uri is not None
    assert expected_path.is_file()
    sink = MilvusEpisodeSink(HashEmbedding(), memory_root=backend.root)
    try:
        assert sink.client.get_collection_stats(
            episode_collection_name(VERSION)
        )["row_count"] == 1
    finally:
        sink.close()


def test_milvus_failure_preserves_canonical_episode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenMilvusClient:
        def __init__(self, *args, **kwargs) -> None:
            raise OSError("Milvus unavailable")

    monkeypatch.setattr(
        "rpent.episode.milvus_sink.MilvusClient", BrokenMilvusClient
    )
    backend = _backend(tmp_path)
    with caplog.at_level(logging.WARNING, logger="rpent.episode"):
        uri = record_episode(
            backend,
            **_record_kwargs(backend, _run_directory(tmp_path)),
            embedding_client=HashEmbedding(),
        )

    assert uri is not None
    episode_path = (
        backend.episodes_dir
        / VERSION
        / "libero10_swap_t2_s0_20261110T031500.json"
    )
    assert episode_path.is_file()
    assert "canonical episode remains" in caplog.text
    assert "Milvus unavailable" in caplog.text


def test_rebuild_tool_restores_identical_query_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = _backend(tmp_path)
    embedding = LocalMockEmbedding(backend.root)
    episodes = []
    for index in range(5):
        frame_uri = backend.put_frame(f"frame-{index}".encode())
        episode = _episode(f"episode-{index}", first_frame=frame_uri)
        episodes.append((episode, _put_episode(backend, episode)))
    sink = MilvusEpisodeSink(embedding, memory_root=backend.root)
    sink.upsert_many(episodes)
    before = _search(sink, episodes[0][0], embedding)
    sink.close()

    shutil.rmtree(backend.root / ".milvus")
    rebuild_main(
        [
            "--robot",
            "libero",
            "--memory-version",
            VERSION,
            "--memory-dir",
            str(tmp_path / "memory"),
            "--mock",
        ]
    )

    sink = MilvusEpisodeSink(embedding, memory_root=backend.root)
    try:
        after = _search(sink, episodes[0][0], embedding)
        assert after == before
        assert sink.client.get_collection_stats(
            episode_collection_name(VERSION)
        )["row_count"] == len(episodes)
    finally:
        sink.close()
    assert f"Rebuilt {len(episodes)} episodes" in capsys.readouterr().out


def test_milvus_cli_defaults_off_and_exploration_starts_embedding() -> None:
    assert _build_argparser().parse_args([]).milvus_uri is None
    assert _embedding_requested(
        SimpleNamespace(
            record_episodes=False,
            explore=True,
            episode_search_enabled=False,
            enable_episode_search=False,
        )
    )
