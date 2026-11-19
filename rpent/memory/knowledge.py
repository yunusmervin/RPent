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

"""Rebuildable text embeddings for published knowledge leaves."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pymilvus import DataType, MilvusClient

from rpent.episode.milvus_sink import VECTOR_DIM, EpisodeEncoder, default_milvus_uri
from rpent.memory.freeze_state import assert_corpus_writable
from rpent.memory.manager import _canonical_id, _split_frontmatter, _validate
from rpent.utils.config import EMBEDDING_MODEL_ID

_SCOPES = {"global", "task-family"}


def knowledge_collection_name(memory_version: str) -> str:
    """Return the versioned collection name for knowledge leaves."""
    if not memory_version:
        raise ValueError("memory_version must not be empty")
    return f"leaves__{re.sub(r'[^A-Za-z0-9_]', '_', memory_version)}"


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise PermissionError(f"knowledge leaf escapes robot memory root: {path}")
    return resolved


def _is_published(root: Path, path: Path) -> bool:
    parts = path.relative_to(root).parts
    return len(parts) == 2 and parts[0] in _SCOPES and path.suffix == ".md"


@dataclass(frozen=True)
class KnowledgeLeaf:
    """A validated canonical leaf and its bounded embedding input."""

    leaf_id: str
    canonical_id: str
    robot: str
    scope: str
    kind: str
    suite: str
    confidence: str
    leaf_uri: str
    embedding_text: str


def parse_leaf(memory_root: str | Path, path: str | Path, robot: str) -> KnowledgeLeaf:
    """Read a published leaf without crossing the current robot's corpus."""
    root = Path(memory_root).resolve()
    resolved = _resolve_path(root, path)
    if not _is_published(root, resolved):
        raise ValueError(f"not a published global/task-family leaf: {path}")
    metadata, body = _split_frontmatter(resolved)
    # Deprecated published knowledge retains its original searchable status.
    # The merge validator predates the explicit contradicted status.
    validation_metadata = dict(metadata)
    if validation_metadata.get("confidence") == "contradicted":
        validation_metadata["confidence"] = "single-shot"
    _validate(validation_metadata)
    if metadata["scope"] != resolved.parent.name:
        raise ValueError(f"knowledge scope does not match its path: {path}")
    canonical_id = _canonical_id(resolved, metadata)
    if canonical_id != resolved.stem:
        raise ValueError(f"knowledge ID does not match its published path: {path}")
    first_paragraph = re.split(r"\n\s*\n", body.strip(), maxsplit=1)[0][:1000]
    text = (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        + "---\n"
        + first_paragraph
    )
    return KnowledgeLeaf(
        leaf_id=f"{robot}:{canonical_id}",
        canonical_id=canonical_id,
        robot=robot,
        scope=metadata["scope"],
        kind=str(metadata.get("kind") or ""),
        suite=str(metadata.get("suite") or ""),
        confidence=(
            "contradicted"
            if (metadata.get("evidence") or {}).get("contradicted_by")
            else metadata["confidence"]
        ),
        leaf_uri=str(resolved),
        embedding_text=text,
    )


class MilvusKnowledgeSink:
    """Encode published leaves into a derived, robot-isolated collection."""

    def __init__(
        self,
        embedding_client: EpisodeEncoder,
        *,
        memory_root: str | Path,
        robot: str,
        memory_version: str,
        milvus_uri: str | None = None,
        client: MilvusClient | None = None,
    ) -> None:
        if not robot or len(robot) > 64 or any(char in robot for char in "/\\"):
            raise ValueError("robot must be a nonempty robot identifier")
        self._root = Path(memory_root).resolve()
        if not self._root.is_dir():
            raise FileNotFoundError(
                f"knowledge corpus directory not found: {self._root}"
            )
        self._robot = robot
        self._embedding_client = embedding_client
        self.collection_name = knowledge_collection_name(memory_version)
        if client is None:
            assert_corpus_writable(self._root)
            client = MilvusClient(uri=milvus_uri or default_milvus_uri(self._root))
        self._client = client

    @property
    def client(self) -> MilvusClient:
        return self._client

    def close(self) -> None:
        self._client.close()

    def _assert_writable(self) -> None:
        assert_corpus_writable(self._root)
        description = self._client.describe_collection(
            collection_name=self.collection_name
        )
        properties = description.get("properties") or {}
        if str(properties.get("frozen", "false")).lower() == "true":
            raise RuntimeError(f"Milvus collection {self.collection_name!r} is frozen")
        if properties.get("embedding_model") != EMBEDDING_MODEL_ID:
            raise ValueError(
                f"collection {self.collection_name!r} does not use embedding model "
                f"{EMBEDDING_MODEL_ID!r}"
            )

    def _ensure_collection(self) -> None:
        assert_corpus_writable(self._root)
        if self._client.has_collection(collection_name=self.collection_name):
            self._assert_writable()
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="leaf_id",
            datatype=DataType.VARCHAR,
            max_length=512,
            is_primary=True,
        )
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM
        )
        for field in ("robot", "scope", "kind", "suite", "confidence"):
            schema.add_field(
                field_name=field,
                datatype=DataType.VARCHAR,
                max_length=64,
                is_partition_key=field == "robot",
            )
        schema.add_field(
            field_name="leaf_uri", datatype=DataType.VARCHAR, max_length=4096
        )
        indexes = self._client.prepare_index_params()
        indexes.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        for field in ("robot", "scope", "kind", "suite", "confidence"):
            indexes.add_index(field_name=field, index_type="INVERTED")
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
            properties={"embedding_model": EMBEDDING_MODEL_ID},
        )

    def _load_leaves(self, paths: Iterable[str | Path]) -> list[KnowledgeLeaf]:
        leaves: dict[str, KnowledgeLeaf] = {}
        for path in paths:
            resolved = _resolve_path(self._root, path)
            # Inbox, merged drafts, conflicts and task artifacts are not leaves.
            if not _is_published(self._root, resolved):
                continue
            leaf = parse_leaf(self._root, resolved, self._robot)
            leaves[leaf.leaf_id] = leaf
        return list(leaves.values())

    def _encode_rows(self, leaves: list[KnowledgeLeaf]) -> list[dict[str, Any]]:
        rows = []
        for leaf in leaves:
            response = self._embedding_client.encode(text=leaf.embedding_text)
            if response.get("model") != EMBEDDING_MODEL_ID:
                raise ValueError(
                    f"unexpected knowledge embedding model: {response.get('model')!r}"
                )
            vector = response.get("vector")
            if (
                not isinstance(vector, list)
                or response.get("dim") != VECTOR_DIM
                or len(vector) != VECTOR_DIM
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in vector
                )
            ):
                raise ValueError(
                    f"knowledge embedding must have dimension {VECTOR_DIM}"
                )
            rows.append(
                {
                    "leaf_id": leaf.leaf_id,
                    "vector": [float(value) for value in vector],
                    "robot": leaf.robot,
                    "scope": leaf.scope,
                    "kind": leaf.kind,
                    "suite": leaf.suite,
                    "confidence": leaf.confidence,
                    "leaf_uri": leaf.leaf_uri,
                }
            )
        return rows

    def upsert_paths(self, paths: Iterable[str | Path]) -> int:
        """Upsert only canonical published leaves, preserving deprecated status."""
        assert_corpus_writable(self._root)
        leaves = self._load_leaves(paths)
        if not leaves:
            return 0
        self._ensure_collection()
        rows = self._encode_rows(leaves)
        self._assert_writable()
        self._client.upsert(collection_name=self.collection_name, data=rows)
        return len(rows)

    def rebuild(self) -> int:
        """Replace this robot's derived rows from its canonical Markdown leaves."""
        if not self._root.is_dir():
            raise FileNotFoundError(
                f"knowledge corpus directory not found: {self._root}"
            )
        assert_corpus_writable(self._root)
        paths = sorted(
            path for scope in _SCOPES for path in (self._root / scope).glob("*.md")
        )
        leaves = self._load_leaves(paths)
        self._ensure_collection()
        # Publish replacements before pruning stale rows, preserving old knowledge
        # when either the embedding service or Milvus upsert is unavailable.
        rows = self._encode_rows(leaves)
        self._assert_writable()
        if rows:
            self._client.upsert(collection_name=self.collection_name, data=rows)
        self._assert_writable()
        stale_filter = f"robot == {json.dumps(self._robot)}"
        if rows:
            desired_ids = [row["leaf_id"] for row in rows]
            stale_filter += f" and leaf_id not in {json.dumps(desired_ids)}"
        self._client.delete(
            collection_name=self.collection_name,
            filter=stale_filter,
        )
        return len(rows)


class KnowledgeIndexHook:
    """Open the optional derived index only when a merge publishes leaves."""

    def __init__(
        self,
        embedding_client: EpisodeEncoder,
        *,
        memory_root: str | Path,
        robot: str,
        memory_version: str,
        milvus_uri: str | None = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._memory_root = Path(memory_root).resolve()
        self._robot = robot
        self._memory_version = memory_version
        self._milvus_uri = milvus_uri

    def upsert_paths(self, paths: Iterable[str | Path]) -> int:
        """Index published leaves and close the owned Milvus connection."""
        assert_corpus_writable(self._memory_root)
        sink = MilvusKnowledgeSink(
            self._embedding_client,
            memory_root=self._memory_root,
            robot=self._robot,
            memory_version=self._memory_version,
            milvus_uri=self._milvus_uri,
        )
        try:
            return sink.upsert_paths(paths)
        finally:
            sink.close()
