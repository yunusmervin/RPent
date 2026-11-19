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

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import validate

from rpent.episode.hook import record_episode
from rpent.episode.milvus_sink import (
    VECTOR_DIM,
    MilvusEpisodeSink,
    episode_collection_name,
)
from rpent.episode.model import Episode, EpisodeAssets
from rpent.episode.writer import EpisodeWriter
from rpent.memory.agfs_backend import AGFSBackend
from rpent.memory.freeze import check_freeze, freeze_memory_version
from rpent.utils.config import EMBEDDING_MODEL_ID
from tools.check_freeze import main as check_main
from tools.freeze_memory_version import main as freeze_main

VERSION = "GPT_6_astra_low"
REPO = Path(__file__).resolve().parents[3]


class Encoder:
    def encode(self, text: str = "", image_uris: list[str] | None = None) -> dict:
        return {
            "vector": [1.0] + [0.0] * (VECTOR_DIM - 1),
            "model": EMBEDDING_MODEL_ID,
            "dim": VECTOR_DIM,
        }


@pytest.fixture
def snapshot(tmp_path: Path):
    parent = tmp_path / "memory"
    root = parent / "libero"
    corpus = root / VERSION
    corpus.mkdir(parents=True)
    (corpus / "MEMORY.md").write_text("# Canonical memory\n", encoding="utf-8")
    (corpus / "global").mkdir()
    (corpus / "global" / "strategy.md").write_text(
        "Use the black bowl.\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", VERSION], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=T7 Test",
            "-c",
            "user.email=t7@example.invalid",
            "commit",
            "-qm",
            "Canonical corpus",
        ],
        check=True,
    )
    backend = AGFSBackend(
        root, agfs_endpoint="http://127.0.0.1:1", robot="libero", record_episodes=True
    )
    episode = Episode(
        episode_id="libero_t2_s0",
        cell_tag="libero_t2_s0",
        robot="libero",
        suite="libero_10",
        regime="swap",
        task_id=2,
        seed=0,
        task_language="pick up the black bowl",
        outcome="success",
        confidence="verified",
        memory_ids=[],
        memory_version=VERSION,
        embedding_model=EMBEDDING_MODEL_ID,
        created_at=datetime(2026, 11, 19, tzinfo=UTC),
        stats={"elapsed_s": 1, "tool_calls": 1, "output_tokens": 0},
        assets=EpisodeAssets(first_frame=backend.put_frame(b"fixture-frame")),
    )
    episode_uri = EpisodeWriter(backend).write(episode)
    sink = MilvusEpisodeSink(Encoder(), memory_root=root)
    sink.upsert(episode, episode_uri)
    state = SimpleNamespace(
        parent=parent,
        root=root,
        corpus=corpus,
        backend=backend,
        sink=sink,
        episode=episode,
        uri=episode_uri,
        kwargs={"robot": "libero", "version": VERSION, "memory_dir": parent},
    )
    try:
        yield state
    finally:
        sink.close()
        backend.close()
        # pytest must be able to remove the intentionally readonly fixture.
        for path in (root, *root.rglob("*")):
            if path.exists():
                path.chmod(path.stat().st_mode | 0o700)


def test_freeze_manifest_schema_and_canonical_bytes_are_preserved(snapshot) -> None:
    before = {
        path: path.read_bytes() for path in snapshot.corpus.rglob("*") if path.is_file()
    }
    path = freeze_memory_version(**snapshot.kwargs)
    manifest = json.loads(path.read_text())
    validate(
        manifest,
        {
            "type": "object",
            "required": [
                "version",
                "corpus_revision",
                "milvus_collection",
                "embedding_model",
                "frozen_at",
            ],
            "properties": {
                "version": {"const": VERSION},
                "corpus_revision": {"type": "string", "pattern": "^git:[0-9a-f]{40}$"},
                "milvus_collection": {"const": episode_collection_name(VERSION)},
                "embedding_model": {"const": EMBEDDING_MODEL_ID},
                "frozen_at": {"type": "string", "pattern": "Z$"},
            },
        },
    )
    assert {path: path.read_bytes() for path in before} == before
    assert check_freeze(**snapshot.kwargs) == []
    assert freeze_memory_version(**snapshot.kwargs) == path
    assert json.loads(path.read_text()) == manifest


def test_freeze_rejects_cached_and_reopened_vector_writers(snapshot) -> None:
    freeze_memory_version(**snapshot.kwargs)
    with pytest.raises(RuntimeError, match="frozen"):
        snapshot.sink.upsert(snapshot.episode, snapshot.uri)
    reopened = MilvusEpisodeSink(Encoder(), memory_root=snapshot.root)
    try:
        with pytest.raises(RuntimeError, match="frozen"):
            reopened.upsert(snapshot.episode, snapshot.uri)
        assert (
            reopened.client.get_collection_stats(episode_collection_name(VERSION))[
                "row_count"
            ]
            == 1
        )
    finally:
        reopened.close()
    with pytest.raises(PermissionError, match="frozen"):
        EpisodeWriter(snapshot.backend).write(snapshot.episode)


def test_check_freeze_cli_exit_codes(snapshot) -> None:
    args = [
        "--robot",
        "libero",
        "--version",
        VERSION,
        "--memory-dir",
        str(snapshot.parent),
    ]
    assert check_main(args) == 1
    assert freeze_main(args) == 0
    assert check_main(args) == 0
    snapshot.sink.close()
    from milvus_lite.server_manager import server_manager_instance

    server_manager_instance.release_server(
        str(snapshot.root / ".milvus" / "episodes.db")
    )
    for command, code in (
        (args, 0),
        (
            [
                "--robot",
                "libero",
                "--version",
                "absent",
                "--memory-dir",
                str(snapshot.parent),
            ],
            1,
        ),
        ([], 2),
    ):
        completed = subprocess.run(
            [sys.executable, str(REPO / "tools" / "check_freeze.py"), *command],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == code, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "damage", ["body", "permission", "properties", "config", "model", "manifest"]
)
def test_check_freeze_detects_actual_state_drift(snapshot, damage: str) -> None:
    manifest = freeze_memory_version(**snapshot.kwargs)
    if damage in {"body", "permission"}:
        path = snapshot.corpus / "MEMORY.md"
        path.chmod(0o644)
        if damage == "body":
            path.write_text("Changed canonical memory\n")
            path.chmod(0o444)
    elif damage in {"properties", "model"}:
        snapshot.sink.client.alter_collection_properties(
            episode_collection_name(VERSION),
            properties={"frozen": "false"}
            if damage == "properties"
            else {"embedding_model": "different"},
        )
    elif damage == "config":
        (snapshot.root / "_internal" / "freeze" / "agfs-readonly.json").unlink()
    else:
        manifest.write_text('{"version": "wrong"}')
    assert check_freeze(**snapshot.kwargs)


def test_freeze_rejects_uncommitted_corpus_before_mutation(snapshot) -> None:
    (snapshot.corpus / "MEMORY.md").write_text("uncommitted prose")
    with pytest.raises(ValueError, match="uncommitted|differs"):
        freeze_memory_version(**snapshot.kwargs)
    assert (
        "frozen"
        not in snapshot.sink.client.describe_collection(
            episode_collection_name(VERSION)
        )["properties"]
    )
    assert not (snapshot.root / "_internal" / "freeze").exists()


def test_freeze_rejects_deleted_committed_file(snapshot) -> None:
    (snapshot.corpus / "global" / "strategy.md").unlink()
    with pytest.raises(ValueError, match="file list differs"):
        freeze_memory_version(**snapshot.kwargs)


def test_freeze_rejects_index_with_unrelated_rows(snapshot) -> None:
    row = snapshot.sink.client.query(
        episode_collection_name(VERSION),
        filter='episode_id == "libero_t2_s0"',
        output_fields=["*"],
    )[0]
    row["task_id"] = 999
    snapshot.sink.client.upsert(episode_collection_name(VERSION), data=[row])
    with pytest.raises(ValueError, match="rows differ from canonical"):
        freeze_memory_version(**snapshot.kwargs)


def test_check_freeze_detects_vector_tampering_with_same_row_count(snapshot) -> None:
    freeze_memory_version(**snapshot.kwargs)
    collection = episode_collection_name(VERSION)
    snapshot.sink.client.load_collection(collection)
    row = snapshot.sink.client.query(
        collection, filter='episode_id == "libero_t2_s0"', output_fields=["*"]
    )[0]
    row["vector"] = [0.0, 1.0] + [0.0] * (VECTOR_DIM - 2)
    snapshot.sink.client.upsert(collection, data=[row])
    assert "Milvus row content differs from manifest" in check_freeze(**snapshot.kwargs)


def test_check_freeze_detects_writable_empty_directory(snapshot) -> None:
    empty = snapshot.corpus / "empty"
    empty.mkdir()
    freeze_memory_version(**snapshot.kwargs)
    empty.chmod(0o755)
    assert any("writable" in error for error in check_freeze(**snapshot.kwargs))


def test_explicit_recording_refuses_frozen_version_before_extracting_frames(
    snapshot, caplog: pytest.LogCaptureFixture
) -> None:
    freeze_memory_version(**snapshot.kwargs)
    before = sorted(snapshot.backend.frames_dir.rglob("*.png"))
    assert (
        record_episode(
            snapshot.backend,
            run_state_dir=snapshot.root / "nonexistent-run",
            cell_tag="libero_t2_s0",
            robot="libero",
            task_desc={},
            finish_result={},
            stats={},
            elapsed_s=0,
            memory_version=VERSION,
            allow_read_only_write=True,
            embedding_client=Encoder(),
        )
        is None
    )
    assert "frozen" in caplog.text
    assert sorted(snapshot.backend.frames_dir.rglob("*.png")) == before


def test_freeze_requires_existing_collection_without_creating_database(
    snapshot,
) -> None:
    missing = snapshot.root / "missing.db"
    with pytest.raises(ValueError, match="database does not exist"):
        freeze_memory_version(**snapshot.kwargs, milvus_uri=str(missing))
    assert not missing.exists()
    snapshot.sink.client.drop_collection(episode_collection_name(VERSION))
    with pytest.raises(ValueError, match="collection does not exist"):
        freeze_memory_version(**snapshot.kwargs)


def test_freeze_rejects_cross_robot_and_symlink_paths(snapshot) -> None:
    other = snapshot.parent / "robotwin"
    other.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        freeze_memory_version(**snapshot.kwargs, corpus_dir=other)
    (snapshot.corpus / "escape.md").symlink_to(other / "foreign.md")
    with pytest.raises(ValueError, match="symlink"):
        freeze_memory_version(**snapshot.kwargs)


def test_hf_freeze_pins_and_verifies_corpus_revision(
    snapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    commit = "a" * 40
    prefix = f"libero/{VERSION}/"
    files = [path for path in snapshot.corpus.rglob("*") if path.is_file()]

    class API:
        def repo_info(self, repo, *, repo_type, revision):
            assert (repo, repo_type, revision) == ("owner/memory", "dataset", "release")
            return SimpleNamespace(
                sha=commit,
                siblings=[
                    SimpleNamespace(
                        rfilename=prefix + path.relative_to(snapshot.corpus).as_posix()
                    )
                    for path in files
                ],
            )

    def download(repo, filename, *, repo_type, revision):
        assert revision == commit
        return snapshot.corpus / filename.removeprefix(prefix)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    path = freeze_memory_version(**snapshot.kwargs, memory_repo="owner/memory@release")
    assert (
        json.loads(path.read_text())["corpus_revision"] == f"hf:owner/memory@{commit}"
    )
    assert check_freeze(**snapshot.kwargs) == []
