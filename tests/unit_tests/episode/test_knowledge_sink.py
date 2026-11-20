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

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from rpent.memory.freeze_state import freeze_mounts
from rpent.memory.knowledge import (
    VECTOR_DIM,
    KnowledgeIndexHook,
    MilvusKnowledgeSink,
    knowledge_collection_name,
    parse_leaf,
)
from rpent.utils.config import EMBEDDING_MODEL_ID

VERSION = "knowledge_test"


class TextEmbedding:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]:
        self.texts.append(text)
        return {
            "vector": [1.0, *([0.0] * (VECTOR_DIM - 1))],
            "model": EMBEDDING_MODEL_ID,
            "dim": VECTOR_DIM,
        }


def _leaf(
    root: Path,
    name: str = "grasp",
    *,
    scope: str = "global",
    confidence: str = "verified",
    body: str = "Approach the bowl from above.\n\nDo not encode this paragraph.",
) -> Path:
    metadata: dict[str, Any] = {
        "scope": scope,
        "kind": "strategy",
        "title": "Grasp a bowl",
        "applies_when": "The bowl must be relocated",
        "confidence": confidence,
        "evidence": {"cells": ["libero10_swap_t2_s0"]},
    }
    if scope == "task-family":
        name = "task-family_libero_10_swap_t2"
        metadata.update(
            suite="libero_10",
            regime="swap",
            task_id=2,
            task_language="move the bowl",
        )
    metadata["id"] = name
    path = root / scope / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n" + yaml.safe_dump(metadata) + "---\n" + body)
    return path


def _sink(
    root: Path, embedding: TextEmbedding | None = None, **kwargs: Any
) -> MilvusKnowledgeSink:
    return MilvusKnowledgeSink(
        embedding or TextEmbedding(),
        memory_root=root,
        robot="libero",
        memory_version=VERSION,
        **kwargs,
    )


def _rows(sink: MilvusKnowledgeSink) -> list[dict[str, Any]]:
    return sorted(
        sink.client.query(
            collection_name=sink.collection_name,
            filter='leaf_id != ""',
            output_fields=[
                "leaf_id",
                "robot",
                "scope",
                "kind",
                "suite",
                "confidence",
                "leaf_uri",
            ],
        ),
        key=lambda row: row["leaf_id"],
    )


def test_frontmatter_and_first_paragraph_are_encoded_with_bounded_body(
    tmp_path: Path,
) -> None:
    path = _leaf(tmp_path, body="字" * 1100 + "\n\nexcluded paragraph")
    leaf = parse_leaf(tmp_path, path, "libero")
    assert leaf.leaf_id == "libero:grasp"
    assert leaf.canonical_id == "grasp"
    assert leaf.leaf_uri == str(path.resolve())
    assert "confidence: verified" in leaf.embedding_text
    assert leaf.embedding_text.endswith("字" * 1000)
    assert "字" * 1001 not in leaf.embedding_text
    assert "excluded paragraph" not in leaf.embedding_text


def test_schema_upsert_and_deprecated_confidence(tmp_path: Path) -> None:
    path = _leaf(tmp_path, confidence="contradicted")
    family = _leaf(tmp_path, scope="task-family")
    sink = _sink(tmp_path)
    try:
        assert sink.upsert_paths([path, family, path]) == 2
        assert sink.upsert_paths([path]) == 1
        description = sink.client.describe_collection(sink.collection_name)
        fields = {field["name"]: field for field in description["fields"]}
        assert set(fields) == {
            "leaf_id",
            "vector",
            "robot",
            "scope",
            "kind",
            "suite",
            "confidence",
            "leaf_uri",
        }
        assert fields["leaf_id"]["is_primary"] is True
        assert fields["vector"]["params"]["dim"] == VECTOR_DIM
        assert fields["robot"]["is_partition_key"] is True
        assert description["properties"]["embedding_model"] == EMBEDDING_MODEL_ID
        assert (
            sink.client.describe_index(sink.collection_name, "vector")["metric_type"]
            == "COSINE"
        )
        rows = _rows(sink)
        assert len(rows) == 2
        assert rows[0]["confidence"] == "contradicted"
        assert rows[1]["suite"] == "libero_10"
        assert knowledge_collection_name("GPT 6:astra/low") == "leaves__GPT_6_astra_low"
    finally:
        sink.close()


def test_contradiction_evidence_sets_derived_confidence_without_editing_corpus(
    tmp_path: Path,
) -> None:
    path = _leaf(tmp_path)
    original = path.read_text().replace(
        "evidence:\n", "evidence:\n  contradicted_by: [libero10_swap_t2_s1]\n"
    )
    path.write_text(original)
    embedding = TextEmbedding()
    sink = _sink(tmp_path, embedding)
    try:
        sink.upsert_paths([path])
        assert _rows(sink)[0]["confidence"] == "contradicted"
        assert "confidence: verified" in embedding.texts[0]
        assert path.read_text() == original
        assert sink.rebuild() == 1
        assert _rows(sink)[0]["confidence"] == "contradicted"
    finally:
        sink.close()


def test_conflicts_and_internal_drafts_are_not_indexed(tmp_path: Path) -> None:
    path = _leaf(tmp_path)
    conflict = tmp_path / "_internal" / "conflicts" / "grasp.md"
    conflict.parent.mkdir(parents=True)
    conflict.write_text(path.read_text())
    draft = tmp_path / "_internal" / "inbox" / "cell" / "grasp.md"
    draft.parent.mkdir(parents=True)
    draft.write_text(path.read_text())
    embedding = TextEmbedding()
    sink = _sink(tmp_path, embedding)
    try:
        assert sink.upsert_paths([conflict, path, draft]) == 1
        assert len(embedding.texts) == 1
        assert [row["leaf_id"] for row in _rows(sink)] == ["libero:grasp"]
        assert sink.rebuild() == 1
    finally:
        sink.close()


def test_robot_scoped_rebuild_removes_stale_rows_and_preserves_other_robot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "libero"
    other_root = tmp_path / "franka"
    uri = str(tmp_path / "shared.db")
    path = _leaf(root)
    stale = _leaf(root, "stale")
    other = _leaf(other_root)
    sink = _sink(root, milvus_uri=uri)
    other_sink = MilvusKnowledgeSink(
        TextEmbedding(),
        memory_root=other_root,
        robot="franka",
        memory_version=VERSION,
        milvus_uri=uri,
    )
    try:
        sink.upsert_paths([path, stale])
        other_sink.upsert_paths([other])
        stale.unlink()
        _leaf(root, "new")
        assert sink.rebuild() == 2
        before = _rows(sink)
        assert [row["leaf_id"] for row in before] == [
            "franka:grasp",
            "libero:grasp",
            "libero:new",
        ]
        assert sink.rebuild() == 2
        assert _rows(sink) == before
        path.unlink()
        (root / "global" / "new.md").unlink()
        assert sink.rebuild() == 0
        assert [row["leaf_id"] for row in _rows(sink)] == ["franka:grasp"]
    finally:
        sink.close()
        other_sink.close()


def test_escape_and_cross_scope_paths_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "libero"
    outside = _leaf(tmp_path / "franka")
    with pytest.raises(PermissionError, match="escapes robot"):
        parse_leaf(root, outside, "libero")
    link = root / "global" / "grasp.md"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    with pytest.raises(PermissionError, match="escapes robot"):
        parse_leaf(root, link, "libero")
    path = _leaf(root, "wrong-scope")
    path.write_text(path.read_text().replace("scope: global", "scope: task-family"))
    with pytest.raises(ValueError):
        parse_leaf(root, path, "libero")


def test_missing_corpus_cannot_create_a_directory_or_remove_existing_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "libero"
    path = _leaf(root, confidence="contradicted")
    sink = _sink(root, milvus_uri=str(tmp_path / "index.db"))
    try:
        sink.upsert_paths([path])
        before = _rows(sink)
        missing = tmp_path / "missing"
        with pytest.raises(FileNotFoundError, match="corpus directory not found"):
            _sink(missing)
        with pytest.raises(FileNotFoundError, match="corpus directory not found"):
            _sink(missing, client=sink.client)
        assert not missing.exists()
        assert _rows(sink) == before
        shutil.rmtree(root)
        with pytest.raises(FileNotFoundError, match="corpus directory not found"):
            sink.rebuild()
        assert _rows(sink) == before
        root.mkdir()
        assert sink.rebuild() == 0
        assert _rows(sink) == []
    finally:
        sink.close()


def test_collection_freeze_is_checked_by_a_live_sink(tmp_path: Path) -> None:
    path = _leaf(tmp_path)
    sink = _sink(tmp_path)
    try:
        sink.upsert_paths([path])
        sink.client.alter_collection_properties(
            collection_name=sink.collection_name, properties={"frozen": "true"}
        )
        with pytest.raises(RuntimeError, match="frozen"):
            sink.upsert_paths([path])
        with pytest.raises(RuntimeError, match="frozen"):
            sink.rebuild()
        assert len(_rows(sink)) == 1
    finally:
        sink.close()


def test_corpus_freeze_prevents_new_derived_writes(tmp_path: Path) -> None:
    path = _leaf(tmp_path)
    sink = _sink(tmp_path)
    try:
        sink.upsert_paths([path])
        freeze_mounts(
            tmp_path,
            VERSION,
            tmp_path,
            tmp_path / "_internal" / "episodes" / VERSION,
            "2026-11-19T00:00:00Z",
        )
        with pytest.raises(PermissionError, match="frozen"):
            sink.upsert_paths([path])
        with pytest.raises(PermissionError, match="frozen"):
            sink.rebuild()
        assert len(_rows(sink)) == 1
    finally:
        sink.close()


def test_encoding_failure_keeps_old_rows_during_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _leaf(tmp_path)
    embedding = TextEmbedding()
    sink = _sink(tmp_path, embedding)
    try:
        sink.upsert_paths([path])
        before = _rows(sink)

        def broken_encode(**kwargs: Any) -> dict[str, Any]:
            raise OSError("embedding service unavailable")

        monkeypatch.setattr(embedding, "encode", broken_encode)
        _leaf(tmp_path, "new")
        with pytest.raises(OSError, match="unavailable"):
            sink.rebuild()
        assert _rows(sink) == before
    finally:
        sink.close()


def test_upsert_failure_keeps_old_rows_during_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _leaf(tmp_path, confidence="contradicted")
    sink = _sink(tmp_path)
    try:
        sink.upsert_paths([path])
        before = _rows(sink)
        path.unlink()
        _leaf(tmp_path, "replacement")

        def broken_upsert(**kwargs: Any) -> None:
            raise OSError("Milvus unavailable")

        monkeypatch.setattr(sink.client, "upsert", broken_upsert)
        with pytest.raises(OSError, match="Milvus unavailable"):
            sink.rebuild()
        assert _rows(sink) == before
    finally:
        sink.close()


def test_lazy_hook_does_not_open_milvus_until_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeSink:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append("open")

        def upsert_paths(self, paths: Any) -> int:
            calls.append("upsert")
            raise OSError("Milvus unavailable")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr("rpent.memory.knowledge.MilvusKnowledgeSink", FakeSink)
    hook = KnowledgeIndexHook(
        TextEmbedding(),
        memory_root=tmp_path,
        robot="libero",
        memory_version=VERSION,
    )
    assert calls == []
    with pytest.raises(OSError, match="Milvus unavailable"):
        hook.upsert_paths([tmp_path / "global" / "grasp.md"])
    assert calls == ["open", "upsert", "close"]
