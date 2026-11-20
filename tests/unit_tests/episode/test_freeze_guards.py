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

"""Freeze guards must survive live backends and concurrent maintenance."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.episode.milvus_sink import VECTOR_DIM, MilvusEpisodeSink
from rpent.memory.agfs_backend import AGFSBackend
from rpent.memory.freeze_state import (
    assert_corpus_writable,
    freeze_mounts,
    read_freeze_config,
)
from rpent.memory.manager import FileSystemBackend
from rpent.utils.config import EMBEDDING_MODEL_ID

VERSION = "freeze-guards"
FROZEN_AT = "2026-11-19T00:00:00Z"


class _Client:
    frozen = False
    writes = 0
    drops = 0

    def has_collection(self, **kwargs):
        return True

    def describe_collection(self, **kwargs):
        return {
            "properties": {
                "embedding_model": EMBEDDING_MODEL_ID,
                "frozen": str(self.frozen).lower(),
            }
        }

    def upsert(self, **kwargs):
        self.writes += 1

    def drop_collection(self, **kwargs):
        self.drops += 1


class _Encoder:
    calls = 0

    def __init__(self, callback=lambda: None):
        self.callback = callback

    def encode(self, **kwargs):
        self.calls += 1
        self.callback()
        return {
            "model": EMBEDDING_MODEL_ID,
            "vector": [0.0] * VECTOR_DIM,
            "dim": VECTOR_DIM,
        }


def _episode():
    return SimpleNamespace(
        memory_version=VERSION,
        embedding_model=EMBEDDING_MODEL_ID,
        task_language="pick up the bowl",
        assets=SimpleNamespace(first_frame="agfs://frames/first.png"),
        episode_id="one",
        robot="libero",
        suite="libero_10",
        regime="swap",
        task_id=0,
        seed=0,
        outcome="success",
        confidence="verified",
        memory_ids=[],
    )


def test_cached_sink_observes_freeze_before_encoding():
    client = _Client()
    encoder = _Encoder()
    sink = MilvusEpisodeSink(encoder, client=client)
    sink.upsert(_episode(), "agfs://episodes/one.json")
    client.frozen = True

    with pytest.raises(RuntimeError, match="frozen"):
        sink.upsert(_episode(), "agfs://episodes/one.json")
    assert encoder.calls == 1
    assert client.writes == 1


def test_sink_rechecks_freeze_after_encoding():
    client = _Client()
    encoder = _Encoder(lambda: setattr(client, "frozen", True))
    sink = MilvusEpisodeSink(encoder, client=client)

    with pytest.raises(RuntimeError, match="frozen"):
        sink.upsert(_episode(), "agfs://episodes/one.json")
    assert encoder.calls == 1
    assert client.writes == 0


def test_rebuild_cannot_drop_frozen_collection(tmp_path):
    (tmp_path / VERSION).mkdir()
    client = _Client()
    client.frozen = True
    sink = MilvusEpisodeSink(_Encoder(), client=client)

    with pytest.raises(RuntimeError, match="frozen"):
        sink.rebuild(episodes_dir=tmp_path, memory_version=VERSION)
    assert client.drops == 0


def _freeze(root: Path, *, corpus: Path | None = None):
    episodes = root / "_internal/agfs/episodes" / VERSION
    freeze_mounts(root, VERSION, corpus or root, episodes, FROZEN_AT)
    return episodes


def test_existing_agfs_backend_rejects_even_existing_frozen_episode(tmp_path):
    root = tmp_path / "memory/libero"
    backend = AGFSBackend(root, agfs_endpoint="http://unused", record_episodes=True)
    backend.put_episode(memory_version=VERSION, episode_id="one", payload=b"{}")
    episodes = _freeze(root)

    with pytest.raises(PermissionError, match="frozen"):
        backend.put_episode(memory_version=VERSION, episode_id="one", payload=b"{}")
    assert any(
        mount["path"] == f"/episodes/{VERSION}"
        and Path(mount["config"]["local_dir"]) == episodes
        for mount in backend._mounts()
    )


@pytest.mark.parametrize("operation", ["authorize", "merge", "index", "sync"])
def test_frozen_corpus_denies_all_filesystem_mutations(tmp_path, operation):
    root = tmp_path / "memory/libero"
    backend = FileSystemBackend(
        root, memory_access="inbox_write", inbox_cell_tag="cell"
    )
    _freeze(root)

    with pytest.raises(PermissionError, match="frozen"):
        if operation == "authorize":
            backend.authorize_write(root / "_internal/inbox/cell/draft.md")
        elif operation == "merge":
            backend.merge_memory(cell_tag="cell", run_state_dir=tmp_path, solved=False)
        elif operation == "index":
            backend.rebuild_index()
        else:
            backend.sync(remote_repo="unused")
    assert not (root / "_internal/inbox").exists()


def test_nested_corpus_detects_parent_robot_freeze(tmp_path):
    root = tmp_path / "memory/libero"
    corpus = root / VERSION
    _freeze(root, corpus=corpus)

    with pytest.raises(PermissionError, match="frozen"):
        assert_corpus_writable(corpus)
    with pytest.raises(PermissionError, match="frozen"):
        AGFSBackend(
            corpus, agfs_endpoint="http://unused", record_episodes=True
        ).put_episode(memory_version="new", episode_id="one", payload=b"{}")
    assert_corpus_writable(tmp_path / "memory/franka")
    assert_corpus_writable(root / "other-version")


def test_recursive_sync_cannot_overwrite_nested_frozen_corpus(tmp_path):
    root = tmp_path / "memory/libero"
    _freeze(root, corpus=root / VERSION)

    with pytest.raises(PermissionError, match="frozen"):
        FileSystemBackend(root).sync(remote_repo="unused")


def test_corrupt_freeze_configuration_fails_closed(tmp_path):
    root = tmp_path / "memory/libero"
    _freeze(root)
    (root / "_internal/freeze/agfs-readonly.json").write_text("broken")

    with pytest.raises(ValueError, match="invalid freeze configuration"):
        assert_corpus_writable(root)


def test_freeze_rejects_symlink_escape_without_creating_external_files(tmp_path):
    root = tmp_path / "memory/libero"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    (root / "_internal").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        _freeze(root)
    assert list(outside.iterdir()) == []


def test_freeze_metadata_is_idempotent_and_survives_reload(tmp_path):
    root = tmp_path / "memory/libero"
    _freeze(root)
    _freeze(root)
    assert read_freeze_config(root)["versions"][VERSION]["frozen_at"] == FROZEN_AT
