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

"""Read-only semantic retrieval over the derived episode index."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field
from pymilvus import MilvusClient

from rpent.episode.milvus_sink import (
    VECTOR_DIM,
    EpisodeEncoder,
    episode_collection_name,
)
from rpent.episode.model import Episode
from rpent.tools import ToolResult, tool

if TYPE_CHECKING:
    from rpent.tools.toolkit import Toolkit

_OUTPUT_FIELDS = [
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


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _episode_path(episodes_dir: Path, episode_uri: str) -> Path:
    prefix = "agfs://episodes/"
    if not episode_uri.startswith(prefix):
        raise ValueError(f"unsupported episode URI: {episode_uri!r}")
    path = (episodes_dir / episode_uri.removeprefix(prefix)).resolve()
    if not path.is_relative_to(episodes_dir):
        raise ValueError(f"episode URI escapes local storage: {episode_uri!r}")
    return path


class EpisodeSearchTool:
    """Bind one robot and memory version to the episode search declaration."""

    def __init__(
        self,
        *,
        robot: str,
        memory_version: str,
        embedding_client: EpisodeEncoder | None,
        milvus_client: Any,
        episodes_dir: str | Path,
        owns_milvus_client: bool = False,
    ) -> None:
        self._robot = robot
        self._memory_version = memory_version
        self._collection_name = episode_collection_name(memory_version)
        self._embedding_client = embedding_client
        self._milvus_client = milvus_client
        self._episodes_dir = Path(episodes_dir).resolve()
        self._owns_milvus_client = owns_milvus_client

    def close(self) -> None:
        """Close the Milvus client only when this object created it."""
        if self._owns_milvus_client:
            self._milvus_client.close()
            self._owns_milvus_client = False

    def _query_vector(
        self, query_text: str | None, current_frame: str | None
    ) -> list[float]:
        if self._embedding_client is None:
            raise RuntimeError("embedding service is not configured")
        response = self._embedding_client.encode(
            text=query_text or "",
            image_uris=[current_frame] if current_frame else [],
        )
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
                f"episode query embedding must have dimension {VECTOR_DIM}"
            )
        return [float(value) for value in vector]

    def _summary(self, hit: dict[str, Any]) -> dict[str, Any]:
        entity = hit.get("entity") or {}
        summary = dict(entity)
        for key, value in hit.items():
            if key != "entity":
                summary.setdefault(key, value)
        episode_uri = summary["episode_uri"]
        path = _episode_path(self._episodes_dir, episode_uri)
        episode = Episode.model_validate_json(path.read_bytes())
        if (
            episode.episode_id != summary.get("episode_id")
            or episode.robot != self._robot
            or episode.memory_version != self._memory_version
        ):
            raise ValueError(
                f"episode metadata does not match search hit: {episode_uri}"
            )
        summary["task_language"] = episode.task_language
        summary["assets"] = episode.assets.model_dump(mode="json")
        return summary

    @tool(readonly=True)
    def search_episodes(
        self,
        query_text: str | None = None,
        current_frame: str | None = None,
        suite: str | None = None,
        regime: str | None = None,
        outcome: str | None = None,
        top_k: Annotated[int, Field(strict=True, ge=1, le=100)] = 5,
    ) -> ToolResult:
        """Search historical episodes and return summaries, frame URIs, and linked memory IDs. After recall, use read_text_file to read the linked knowledge leaves before making a decision.

        Args:
            query_text: Task language or another semantic text query.
            current_frame: Current first frame as an agfs:// or file:// URI.
            suite: Optional exact suite filter.
            regime: Optional exact regime filter.
            outcome: Optional exact outcome filter.
            top_k: Maximum number of episodes to return (default 5).
        """
        query_text = query_text.strip() if query_text is not None else None
        current_frame = current_frame.strip() if current_frame is not None else None
        if not query_text and not current_frame:
            raise ValueError("query_text or current_frame must be provided")
        try:
            vector = self._query_vector(query_text, current_frame)
            filters = [f"robot == {_quoted(self._robot)}"]
            for field_name, value in (
                ("suite", suite),
                ("regime", regime),
                ("outcome", outcome),
            ):
                if value is not None:
                    filters.append(f"{field_name} == {_quoted(value)}")
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
            return ToolResult(data={"episodes": [self._summary(hit) for hit in hits]})
        except Exception as exc:
            return ToolResult(
                data={
                    "error": "episode_search_unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "robot": self._robot,
                    "memory_version": self._memory_version,
                }
            )


def create_episode_search(
    *,
    robot: str,
    memory_version: str,
    embedding_client: EpisodeEncoder | None,
    memory_root: str | Path,
    milvus_uri: str | None = None,
    milvus_client: Any = None,
    episodes_dir: str | Path | None = None,
) -> EpisodeSearchTool | None:
    """Open a searcher only when the selected derived collection exists."""
    memory_root = Path(memory_root).resolve()
    owns_client = milvus_client is None
    if milvus_client is None:
        uri = milvus_uri or str(memory_root / ".milvus" / "episodes.db")
        if "://" not in uri and not Path(uri).is_file():
            return None
        try:
            milvus_client = MilvusClient(uri=uri)
        except Exception:
            return None
    try:
        if not milvus_client.has_collection(
            collection_name=episode_collection_name(memory_version)
        ):
            if owns_client:
                milvus_client.close()
            return None
    except Exception:
        if owns_client:
            milvus_client.close()
        return None
    return EpisodeSearchTool(
        robot=robot,
        memory_version=memory_version,
        embedding_client=embedding_client,
        milvus_client=milvus_client,
        episodes_dir=episodes_dir or memory_root / "_internal" / "agfs" / "episodes",
        owns_milvus_client=owns_client,
    )


def register_episode_search(
    toolkit: Toolkit,
    *,
    enabled: bool,
    robot: str,
    memory_version: str,
    embedding_client: EpisodeEncoder | None,
    memory_root: str | Path,
    milvus_uri: str | None = None,
    milvus_client: Any = None,
    episodes_dir: str | Path | None = None,
) -> EpisodeSearchTool | None:
    """Register the readonly tool only for an enabled, existing episode index."""
    if not enabled:
        return None
    search = create_episode_search(
        robot=robot,
        memory_version=memory_version,
        embedding_client=embedding_client,
        memory_root=memory_root,
        milvus_uri=milvus_uri,
        milvus_client=milvus_client,
        episodes_dir=episodes_dir,
    )
    if search is not None:
        toolkit.add_tool(search.search_episodes)
    return search
