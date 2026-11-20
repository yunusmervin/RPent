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

"""Read-only retrieval of canonical Markdown leaves from their derived index."""

from __future__ import annotations

import json
import math
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field
from pymilvus import MilvusClient

from rpent.episode.milvus_sink import EpisodeEncoder
from rpent.memory.knowledge import VECTOR_DIM, knowledge_collection_name
from rpent.tools import ToolResult, tool
from rpent.utils.config import EMBEDDING_MODEL_ID

if TYPE_CHECKING:
    from rpent.tools.toolkit import Toolkit

_SCOPES = {"global", "task-family"}
_KINDS = {"primitive", "perception", "strategy", "failure", "infra"}
_OUTPUT_FIELDS = [
    "leaf_id",
    "robot",
    "scope",
    "kind",
    "suite",
    "confidence",
    "leaf_uri",
]


class KnowledgeSearchTool:
    """Bind one robot's canonical corpus and version to a readonly search tool."""

    def __init__(
        self,
        *,
        robot: str,
        memory_version: str,
        embedding_client: EpisodeEncoder | None,
        milvus_client: Any,
        memory_root: str | Path,
        owns_milvus_client: bool = False,
    ) -> None:
        self._robot = robot
        self._memory_version = memory_version
        self._collection_name = knowledge_collection_name(memory_version)
        self._embedding_client = embedding_client
        self._milvus_client = milvus_client
        self._memory_root = Path(memory_root).resolve()
        self._owns_milvus_client = owns_milvus_client

    def close(self) -> None:
        """Close the Milvus client only when this object created it."""
        if self._owns_milvus_client:
            self._milvus_client.close()
            self._owns_milvus_client = False

    def _query_vector(self, query_text: str) -> list[float]:
        if self._embedding_client is None:
            raise RuntimeError("embedding service is not configured")
        description = self._milvus_client.describe_collection(
            collection_name=self._collection_name
        )
        properties = description.get("properties") or {}
        if properties.get("embedding_model") != EMBEDDING_MODEL_ID:
            raise ValueError("knowledge index embedding model does not match")
        # Persisted Lite indexes and frozen collections may be released when
        # opened by a new process; loading enables reads without altering rows.
        self._milvus_client.load_collection(collection_name=self._collection_name)
        response = self._embedding_client.encode(text=query_text, image_uris=[])
        if response.get("model") != properties["embedding_model"]:
            raise ValueError("knowledge query embedding model does not match")
        vector = response.get("vector")
        if (
            not isinstance(vector, list)
            or len(vector) != VECTOR_DIM
            or response.get("dim") != VECTOR_DIM
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            )
        ):
            raise ValueError(
                f"knowledge query embedding must have dimension {VECTOR_DIM} "
                "and finite numbers"
            )
        return [float(value) for value in vector]

    def _summary(
        self, hit: dict[str, Any], *, scope: str | None, kind: str | None
    ) -> dict[str, Any]:
        entity = hit.get("entity") or {}
        summary = {field: entity[field] for field in _OUTPUT_FIELDS}
        if summary["robot"] != self._robot:
            raise ValueError("knowledge search returned another robot's leaf")
        if "id" in hit and hit["id"] != summary["leaf_id"]:
            raise ValueError("knowledge hit ID does not match its indexed leaf")
        if summary["scope"] not in _SCOPES:
            raise ValueError("knowledge search returned an unsupported scope")
        for field_name, expected in (("scope", scope), ("kind", kind)):
            if expected is not None and summary[field_name] != expected:
                raise ValueError("knowledge search returned a leaf outside the filter")
        leaf_uri = summary["leaf_uri"]
        if not isinstance(leaf_uri, str) or not Path(leaf_uri).is_absolute():
            raise ValueError("knowledge leaf URI must be an absolute corpus path")
        path = Path(leaf_uri).resolve()
        if not path.is_relative_to(self._memory_root):
            raise ValueError("knowledge leaf URI escapes the current robot's corpus")
        relative = path.relative_to(self._memory_root)
        if (
            len(relative.parts) != 2
            or relative.parts[0] != summary["scope"]
            or path.suffix != ".md"
            or not path.is_file()
            or summary["leaf_id"] != f"{self._robot}:{path.stem}"
        ):
            raise ValueError("knowledge hit does not identify a published corpus leaf")
        summary["leaf_uri"] = str(path)
        if "distance" in hit:
            distance = hit["distance"]
            if (
                not isinstance(distance, (int, float))
                or isinstance(distance, bool)
                or not math.isfinite(distance)
            ):
                raise ValueError("knowledge search returned an invalid distance")
            summary["distance"] = distance
        return summary

    @tool(readonly=True)
    def search_knowledge(
        self,
        query_text: str,
        scope: str | None = None,
        kind: str | None = None,
        top_k: Annotated[int, Field(strict=True, ge=1, le=100)] = 5,
    ) -> ToolResult:
        """Search published knowledge leaves. Use read_text_file on the returned leaf_uri to read the canonical text and register the read before deciding. If unavailable, read MEMORY.md and use its existing grep path.

        Args:
            query_text: Natural-language knowledge query; must not be blank.
            scope: Optional exact global or task-family scope filter.
            kind: Optional exact global knowledge kind filter.
            top_k: Maximum number of leaves to return (default 5).
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must not be empty")
        query_text = query_text.strip()
        if scope is not None and scope not in _SCOPES:
            raise ValueError("scope must be global or task-family")
        if kind is not None and kind not in _KINDS:
            raise ValueError("kind must be a supported global knowledge kind")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 100
        ):
            raise ValueError("top_k must be an integer between 1 and 100")
        try:
            vector = self._query_vector(query_text)
            filters = [f"robot == {json.dumps(self._robot, ensure_ascii=True)}"]
            for field_name, value in (("scope", scope), ("kind", kind)):
                if value is not None:
                    filters.append(
                        f"{field_name} == {json.dumps(value, ensure_ascii=True)}"
                    )
            matches = self._milvus_client.search(
                collection_name=self._collection_name,
                data=[vector],
                anns_field="vector",
                filter=" and ".join(filters),
                limit=top_k,
                output_fields=_OUTPUT_FIELDS,
                search_params={"metric_type": "COSINE"},
            )
            hits = matches[0] if matches else []
            return ToolResult(
                data={
                    "leaves": [
                        self._summary(hit, scope=scope, kind=kind) for hit in hits
                    ]
                }
            )
        except Exception as exc:
            return ToolResult(
                data={
                    "error": "knowledge_search_unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "robot": self._robot,
                    "memory_version": self._memory_version,
                    "fallback": {
                        "index": str(self._memory_root / "MEMORY.md"),
                        "grep_roots": [
                            str(self._memory_root / directory)
                            for directory in ("global", "task-family")
                        ],
                        "instruction": (
                            "Read MEMORY.md, use the existing grep path, then "
                            "read_text_file on the matching canonical leaves."
                        ),
                    },
                }
            )


def create_knowledge_search(
    *,
    robot: str,
    memory_version: str,
    embedding_client: EpisodeEncoder | None,
    memory_root: str | Path,
    milvus_uri: str | None = None,
    milvus_client: Any = None,
) -> KnowledgeSearchTool | None:
    """Open an existing derived index without creating files or collections."""
    memory_root = Path(memory_root).resolve()
    owns_client = milvus_client is None
    if milvus_client is None:
        uri = milvus_uri or str(memory_root / ".milvus" / "episodes.db")
        if "://" not in uri and not Path(uri).exists():
            return None
        try:
            milvus_client = MilvusClient(uri=uri)
        except Exception:
            return None
    try:
        available = milvus_client.has_collection(
            collection_name=knowledge_collection_name(memory_version)
        )
    except Exception:
        available = False
    if not available:
        if owns_client:
            with suppress(Exception):
                milvus_client.close()
        return None
    return KnowledgeSearchTool(
        robot=robot,
        memory_version=memory_version,
        embedding_client=embedding_client,
        milvus_client=milvus_client,
        memory_root=memory_root,
        owns_milvus_client=owns_client,
    )


def register_knowledge_search(
    toolkit: Toolkit,
    *,
    enabled: bool,
    robot: str,
    memory_version: str,
    embedding_client: EpisodeEncoder | None,
    memory_root: str | Path,
    milvus_uri: str | None = None,
    milvus_client: Any = None,
) -> KnowledgeSearchTool | None:
    """Register knowledge retrieval only when explicitly enabled and indexed."""
    if not enabled:
        return None
    search = create_knowledge_search(
        robot=robot,
        memory_version=memory_version,
        embedding_client=embedding_client,
        memory_root=memory_root,
        milvus_uri=milvus_uri,
        milvus_client=milvus_client,
    )
    if search is not None:
        toolkit.add_tool(search.search_knowledge)
    return search
