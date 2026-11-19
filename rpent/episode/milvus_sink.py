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

"""Derived Milvus index for canonical episode records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from pymilvus import DataType, MilvusClient

from rpent.episode.model import Episode
from rpent.utils.config import EMBEDDING_MODEL_ID

VECTOR_DIM = 1536
MAX_MEMORY_IDS = 64
_SCALAR_INDEXED_FIELDS = (
    "robot",
    "suite",
    "regime",
    "task_id",
    "seed",
    "outcome",
    "confidence",
)


class EpisodeEncoder(Protocol):
    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]: ...


def episode_collection_name(memory_version: str) -> str:
    """Return the collection name for one canonical memory version."""
    if not memory_version:
        raise ValueError("memory_version must not be empty")
    return f"episodes__{re.sub(r'[^A-Za-z0-9_]', '_', memory_version)}"


def default_milvus_uri(memory_root: str | Path) -> str:
    root = Path(memory_root).resolve()
    database_dir = root / ".milvus"
    database_dir.mkdir(parents=True, exist_ok=True)
    return str(database_dir / "episodes.db")


class MilvusEpisodeSink:
    """Encode and idempotently upsert episodes into versioned collections."""

    def __init__(
        self,
        embedding_client: EpisodeEncoder,
        *,
        memory_root: str | Path | None = None,
        milvus_uri: str | None = None,
        client: MilvusClient | None = None,
    ) -> None:
        if client is None:
            if milvus_uri is None:
                if memory_root is None:
                    raise ValueError("memory_root is required for the local Milvus DB")
                milvus_uri = default_milvus_uri(memory_root)
            client = MilvusClient(uri=milvus_uri)
        self._embedding_client = embedding_client
        self._client = client
        self._known_collections: dict[str, str] = {}

    @property
    def client(self) -> MilvusClient:
        return self._client

    def close(self) -> None:
        self._client.close()

    def _ensure_collection(self, memory_version: str, embedding_model: str) -> str:
        collection_name = episode_collection_name(memory_version)
        known_model = self._known_collections.get(collection_name)
        if known_model is not None:
            if known_model != embedding_model:
                raise ValueError(
                    f"collection {collection_name!r} uses embedding model "
                    f"{known_model!r}, not {embedding_model!r}"
                )
            return collection_name
        if self._client.has_collection(collection_name=collection_name):
            description = self._client.describe_collection(
                collection_name=collection_name
            )
            properties = description.get("properties") or {}
            if properties.get("embedding_model") != embedding_model:
                raise ValueError(
                    f"collection {collection_name!r} does not use embedding model "
                    f"{embedding_model!r}"
                )
        else:
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="episode_id",
                datatype=DataType.VARCHAR,
                max_length=128,
                is_primary=True,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=VECTOR_DIM,
            )
            for field_name in ("robot", "suite", "regime"):
                schema.add_field(
                    field_name=field_name,
                    datatype=DataType.VARCHAR,
                    max_length=64,
                    is_partition_key=field_name == "robot",
                )
            for field_name in ("task_id", "seed"):
                schema.add_field(field_name=field_name, datatype=DataType.INT64)
            for field_name in ("outcome", "confidence"):
                schema.add_field(
                    field_name=field_name,
                    datatype=DataType.VARCHAR,
                    max_length=16,
                )
            schema.add_field(
                field_name="memory_ids",
                datatype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=MAX_MEMORY_IDS,
                max_length=128,
            )
            schema.add_field(
                field_name="episode_uri",
                datatype=DataType.VARCHAR,
                max_length=512,
            )
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            for field_name in _SCALAR_INDEXED_FIELDS:
                index_params.add_index(field_name=field_name, index_type="INVERTED")
            self._client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
                properties={"embedding_model": embedding_model},
            )
        self._known_collections[collection_name] = embedding_model
        return collection_name

    def _encode(self, episode: Episode) -> list[float]:
        response = self._embedding_client.encode(
            text=episode.task_language,
            image_uris=[episode.assets.first_frame],
        )
        if response.get("model") != episode.embedding_model:
            raise ValueError(
                f"embedding model mismatch for episode {episode.episode_id!r}: "
                f"{response.get('model')!r} != {episode.embedding_model!r}"
            )
        vector = response.get("vector")
        if (
            not isinstance(vector, list)
            or response.get("dim") != VECTOR_DIM
            or len(vector) != VECTOR_DIM
        ):
            raise ValueError(
                f"episode embedding must have dimension {VECTOR_DIM}"
            )
        return [float(value) for value in vector]

    @staticmethod
    def _row(
        episode: Episode, episode_uri: str, vector: list[float]
    ) -> dict[str, Any]:
        return {
            "episode_id": episode.episode_id,
            "vector": vector,
            "robot": episode.robot,
            "suite": episode.suite,
            "regime": episode.regime,
            "task_id": episode.task_id,
            "seed": episode.seed,
            "outcome": episode.outcome,
            "confidence": episode.confidence,
            "memory_ids": episode.memory_ids,
            "episode_uri": episode_uri,
        }

    def upsert(self, episode: Episode, episode_uri: str) -> None:
        self.upsert_many([(episode, episode_uri)])

    def upsert_many(
        self,
        episodes: list[tuple[Episode, str]],
        *,
        batch_size: int = 100,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not episodes:
            return
        collection_names = {
            episode_collection_name(episode.memory_version) for episode, _ in episodes
        }
        embedding_models = {episode.embedding_model for episode, _ in episodes}
        if len(collection_names) != 1 or len(embedding_models) != 1:
            raise ValueError("one upsert batch must target one collection and model")
        collection_name = self._ensure_collection(
            episodes[0][0].memory_version,
            episodes[0][0].embedding_model,
        )
        rows = [
            self._row(episode, episode_uri, self._encode(episode))
            for episode, episode_uri in episodes
        ]
        for start in range(0, len(rows), batch_size):
            self._client.upsert(
                collection_name=collection_name,
                data=rows[start : start + batch_size],
            )

    def rebuild(
        self,
        *,
        episodes_dir: str | Path,
        memory_version: str,
    ) -> int:
        version_dir = Path(episodes_dir) / memory_version
        paths = sorted(version_dir.glob("*.json"))
        if not version_dir.is_dir():
            raise FileNotFoundError(f"episode version directory not found: {version_dir}")
        episodes: list[tuple[Episode, str]] = []
        for path in paths:
            episode = Episode.model_validate_json(path.read_bytes())
            if episode.memory_version != memory_version or episode.episode_id != path.stem:
                raise ValueError(f"episode metadata does not match its path: {path}")
            episode_uri = (
                f"agfs://episodes/{memory_version}/{episode.episode_id}.json"
            )
            episodes.append((episode, episode_uri))

        collection_name = episode_collection_name(memory_version)
        if self._client.has_collection(collection_name=collection_name):
            self._client.drop_collection(collection_name=collection_name)
        self._known_collections.pop(collection_name, None)
        if episodes:
            self.upsert_many(episodes)
        else:
            self._ensure_collection(memory_version, EMBEDDING_MODEL_ID)
        return len(episodes)


def rebuild_episode_index(
    *,
    episodes_dir: str | Path,
    memory_version: str,
    embedding_client: EpisodeEncoder,
    memory_root: str | Path | None = None,
    milvus_uri: str | None = None,
) -> int:
    """Replace one derived collection from its canonical episode JSON files."""
    sink = MilvusEpisodeSink(
        embedding_client,
        memory_root=memory_root,
        milvus_uri=milvus_uri,
    )
    try:
        return sink.rebuild(
            episodes_dir=episodes_dir,
            memory_version=memory_version,
        )
    finally:
        sink.close()
