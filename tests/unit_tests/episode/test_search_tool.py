# Copyright 2026 The RPent Authors.
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
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.episode.milvus_sink import VECTOR_DIM, MilvusEpisodeSink
from rpent.episode.model import Episode, EpisodeAssets, EpisodeStats
from rpent.episode.search_tool import (
    EpisodeSearchTool,
    create_episode_search,
    register_episode_search,
)
from rpent.memory import MemoryManager
from rpent.tools import Toolkit
from rpent.utils.config import EMBEDDING_MODEL_ID

VERSION = "GPT_6_astra_low"
CREATED = datetime(2026, 11, 10, 3, 15, tzinfo=UTC)


def _episode(
    episode_id: str,
    *,
    task_language: str = "put the blue plate in the cardboard box",
    robot: str = "libero",
    outcome: str = "success",
) -> Episode:
    digest = hashlib.sha256(episode_id.encode()).hexdigest()
    return Episode(
        episode_id=episode_id,
        cell_tag="libero_object_task_t0_s0",
        robot=robot,
        suite="libero_object_task",
        regime="task",
        task_id=0,
        seed=0,
        task_language=task_language,
        outcome=outcome,
        confidence="verified" if outcome == "success" else "single-shot",
        memory_ids=["task-family_libero_object_task_t0"],
        memory_version=VERSION,
        embedding_model=EMBEDDING_MODEL_ID,
        created_at=CREATED,
        stats=EpisodeStats(elapsed_s=1, tool_calls=2, output_tokens=3),
        assets=EpisodeAssets(
            first_frame=f"agfs://frames/sha256/{digest[:2]}/{digest[2:4]}/{digest}.png"
        ),
    )


def _write_episode(episodes_dir: Path, episode: Episode) -> str:
    path = episodes_dir / VERSION / f"{episode.episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(episode.model_dump_json(), encoding="utf-8")
    return f"agfs://episodes/{VERSION}/{episode.episode_id}.json"


class _Embedding:
    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]:
        return {
            "vector": [0.0] * VECTOR_DIM,
            "model": EMBEDDING_MODEL_ID,
            "dim": VECTOR_DIM,
        }


class _RecordingEmbedding(_Embedding):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]:
        self.calls.append({"text": text, "image_uris": image_uris or []})
        return super().encode(text, image_uris)


class _Milvus:
    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []
        self.search_calls: list[dict[str, Any]] = []
        self.collection_exists = True

    def has_collection(self, *, collection_name: str) -> bool:
        return self.collection_exists

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.search_calls.append(kwargs)
        return [self.hits]


def _searcher(
    tmp_path: Path,
    *,
    embedding: Any = None,
    milvus: _Milvus | None = None,
) -> EpisodeSearchTool:
    return EpisodeSearchTool(
        robot="libero",
        memory_version=VERSION,
        embedding_client=embedding or _Embedding(),
        milvus_client=milvus or _Milvus(),
        episodes_dir=tmp_path / "episodes",
    )


@pytest.mark.parametrize(
    ("query_text", "current_frame"),
    [(None, None), ("", None), (None, "  "), (" ", "")],
)
def test_query_requires_text_or_frame(
    tmp_path: Path, query_text: str | None, current_frame: str | None
) -> None:
    with pytest.raises(ValueError, match="query_text or current_frame"):
        _searcher(tmp_path).search_episodes(
            query_text=query_text, current_frame=current_frame
        )


def test_filters_metadata_and_canonical_summary_without_image_bytes(
    tmp_path: Path,
) -> None:
    episode = _episode("blue-plate")
    episode_uri = _write_episode(tmp_path / "episodes", episode)
    hit = {
        "id": episode.episode_id,
        "distance": 0.125,
        "entity": {
            "episode_id": episode.episode_id,
            "robot": episode.robot,
            "suite": episode.suite,
            "regime": episode.regime,
            "task_id": episode.task_id,
            "seed": episode.seed,
            "outcome": episode.outcome,
            "confidence": episode.confidence,
            "memory_ids": episode.memory_ids,
            "episode_uri": episode_uri,
            "backend_metadata": "preserved",
        },
    }
    embedding = _RecordingEmbedding()
    milvus = _Milvus([hit])

    result = _searcher(tmp_path, embedding=embedding, milvus=milvus).search_episodes(
        query_text="  place the blue plate in a box  ",
        current_frame="file:///tmp/current.png",
        suite="libero_object_task",
        regime="task",
        outcome="success",
        top_k=3,
    )

    assert embedding.calls == [
        {
            "text": "place the blue plate in a box",
            "image_uris": ["file:///tmp/current.png"],
        }
    ]
    call = milvus.search_calls[0]
    assert call["filter"] == (
        'robot == "libero" and suite == "libero_object_task" '
        'and regime == "task" and outcome == "success"'
    )
    assert call["limit"] == 3
    assert call["anns_field"] == "vector"
    assert call["output_fields"] == [
        "episode_id",
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
    assert result.images == []
    assert not result.is_error
    summary = result.data["episodes"][0]
    assert summary["episode_id"] == "blue-plate"
    assert summary["episode_uri"] == episode_uri
    assert summary["memory_ids"] == ["task-family_libero_object_task_t0"]
    assert summary["task_language"] == episode.task_language
    assert summary["assets"]["first_frame"] == episode.assets.first_frame
    assert summary["backend_metadata"] == "preserved"
    assert summary["distance"] == 0.125

    def assert_no_bytes(value: Any) -> None:
        assert not isinstance(value, (bytes, bytearray, memoryview))
        if isinstance(value, dict):
            for child in value.values():
                assert_no_bytes(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_bytes(child)

    assert_no_bytes(result.data)


def test_embedding_and_milvus_failures_return_structured_errors(
    tmp_path: Path,
) -> None:
    class BrokenEmbedding:
        def encode(self, **kwargs: Any) -> dict[str, Any]:
            raise OSError("embedding offline")

    embedding_failure = _searcher(tmp_path, embedding=BrokenEmbedding())
    embedding_result = embedding_failure.search_episodes(query_text="plate")
    assert embedding_result.data["error"] == "episode_search_unavailable"
    assert "embedding offline" in embedding_result.data["detail"]

    class BrokenMilvus(_Milvus):
        def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
            raise OSError("Milvus offline")

    milvus_result = _searcher(tmp_path, milvus=BrokenMilvus()).search_episodes(
        current_frame="file:///tmp/current.png"
    )
    assert milvus_result.data["error"] == "episode_search_unavailable"
    assert "Milvus offline" in milvus_result.data["detail"]

    no_embedding = EpisodeSearchTool(
        robot="libero",
        memory_version=VERSION,
        embedding_client=None,
        milvus_client=_Milvus(),
        episodes_dir=tmp_path / "episodes",
    ).search_episodes(query_text="plate")
    assert no_embedding.data["error"] == "episode_search_unavailable"


def test_registration_requires_opt_in_and_existing_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rpent.utils.templates.default_variables", lambda: {"output_dir": "/offline"}
    )
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory" / "libero"),
    )
    milvus = _Milvus()
    kwargs = {
        "robot": "libero",
        "memory_version": VERSION,
        "embedding_client": _Embedding(),
        "memory_root": tmp_path / "memory" / "libero",
        "milvus_client": milvus,
        "episodes_dir": tmp_path / "episodes",
    }

    assert register_episode_search(toolkit, enabled=False, **kwargs) is None
    assert "search_episodes" not in {tool.name for tool in toolkit.list_tools()}

    milvus.collection_exists = False
    assert register_episode_search(toolkit, enabled=True, **kwargs) is None
    assert "search_episodes" not in {tool.name for tool in toolkit.list_tools()}

    milvus.collection_exists = True
    search = register_episode_search(toolkit, enabled=True, **kwargs)
    assert search is not None
    declaration = next(
        tool for tool in toolkit.list_tools() if tool.name == "search_episodes"
    )
    assert declaration.readonly is True
    toolkit.close()


def test_missing_local_index_is_not_created(tmp_path: Path) -> None:
    memory_root = tmp_path / "memory" / "libero"
    search = create_episode_search(
        robot="libero",
        memory_version=VERSION,
        embedding_client=_Embedding(),
        memory_root=memory_root,
    )
    assert search is None
    assert not (memory_root / ".milvus").exists()


class _SemanticEmbedding:
    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]:
        vector = [0.0] * VECTOR_DIM
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:2], "big") % VECTOR_DIM] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return {
            "vector": vector,
            "model": EMBEDDING_MODEL_ID,
            "dim": VECTOR_DIM,
        }


def test_twenty_episode_index_returns_blue_plate_success_first(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "memory" / "libero"
    episodes_dir = memory_root / "_internal" / "agfs" / "episodes"
    embedding = _SemanticEmbedding()
    target = _episode("blue-plate-success")
    episodes = [target]
    for index in range(19):
        episodes.append(
            _episode(
                f"distractor-{index}",
                task_language=(
                    f"move the object{index} beside container{index} "
                    "and away from marker"
                ),
                outcome="failure",
            )
        )
    rows = [(episode, _write_episode(episodes_dir, episode)) for episode in episodes]
    sink = MilvusEpisodeSink(embedding, memory_root=memory_root)
    try:
        sink.upsert_many(rows)
        search = EpisodeSearchTool(
            robot="libero",
            memory_version=VERSION,
            embedding_client=embedding,
            milvus_client=sink.client,
            episodes_dir=episodes_dir,
        )
        result = search.search_episodes(
            query_text="put the blue plate into the cardboard box",
            current_frame="file:///tmp/new-scene-first-frame.png",
            top_k=5,
        )
        assert not result.is_error, json.dumps(result.data)
        assert len(result.data["episodes"]) == 5
        assert result.data["episodes"][0]["episode_id"] == "blue-plate-success"
        assert result.data["episodes"][0]["outcome"] == "success"
    finally:
        sink.close()
