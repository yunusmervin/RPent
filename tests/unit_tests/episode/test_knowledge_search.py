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

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rpent.episode.milvus_sink import VECTOR_DIM
from rpent.memory.knowledge_search import (
    KnowledgeSearchTool,
    create_knowledge_search,
    register_knowledge_search,
)
from rpent.utils.config import EMBEDDING_MODEL_ID

VERSION = "v1"


class _Embedding:
    def __init__(self, **response: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = {
            "vector": [0.1] * VECTOR_DIM,
            "model": EMBEDDING_MODEL_ID,
            "dim": VECTOR_DIM,
            **response,
        }

    def encode(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


class _Milvus:
    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []
        self.search_calls: list[dict[str, Any]] = []
        self.collection_calls: list[dict[str, Any]] = []
        self.collection_exists = True
        self.properties = {"embedding_model": EMBEDDING_MODEL_ID}
        self.closed = 0

    def has_collection(self, **kwargs: Any) -> bool:
        self.collection_calls.append(kwargs)
        return self.collection_exists

    def describe_collection(self, **kwargs: Any) -> dict[str, Any]:
        return {"properties": self.properties}

    def load_collection(self, **kwargs: Any) -> None:
        pass

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.search_calls.append(kwargs)
        return [self.hits]

    def close(self) -> None:
        self.closed += 1


class _Toolkit:
    def __init__(self) -> None:
        self.tools: list[Any] = []

    def add_tool(self, declaration: Any) -> None:
        self.tools.append(declaration)


def _searcher(
    memory_root: Path,
    *,
    embedding: Any = None,
    milvus: Any = None,
) -> KnowledgeSearchTool:
    return KnowledgeSearchTool(
        robot="libero",
        memory_version=VERSION,
        embedding_client=embedding or _Embedding(),
        milvus_client=milvus or _Milvus(),
        memory_root=memory_root,
    )


def _hit(memory_root: Path, **fields: Any) -> dict[str, Any]:
    path = memory_root / "global" / "side-grasp.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("canonical knowledge must be read explicitly", encoding="utf-8")
    return {
        "id": "libero:side-grasp",
        "distance": 0.91,
        "entity": {
            "leaf_id": "libero:side-grasp",
            "robot": "libero",
            "scope": "global",
            "kind": "strategy",
            "suite": "",
            "confidence": "verified",
            "leaf_uri": str(path),
            **fields,
        },
    }


def test_knowledge_search_declaration_and_input_validation(tmp_path: Path) -> None:
    declaration = _searcher(tmp_path).search_knowledge
    assert declaration.readonly is True
    assert declaration.name == "search_knowledge"
    assert declaration.input_schema["required"] == ["query_text"]
    assert set(declaration.input_schema["properties"]) == {
        "query_text",
        "scope",
        "kind",
        "top_k",
    }
    assert "read_text_file" in declaration.description
    for value in (0, 101, True, 1.5, "5"):
        with pytest.raises(ValidationError):
            declaration.args_schema.model_validate(
                {"query_text": "pick from the side", "top_k": value}, strict=True
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_text": "  "},
        {"query_text": "side grasp", "scope": "task-specific"},
        {"query_text": "side grasp", "kind": "unknown"},
        {"query_text": "side grasp", "top_k": 0},
        {"query_text": "side grasp", "top_k": 101},
        {"query_text": "side grasp", "top_k": True},
        {"query_text": "side grasp", "top_k": 1.5},
    ],
)
def test_invalid_queries_do_not_contact_services(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    embedding = _Embedding()
    milvus = _Milvus()
    with pytest.raises(ValueError):
        _searcher(tmp_path, embedding=embedding, milvus=milvus).search_knowledge(
            **kwargs
        )
    assert not embedding.calls
    assert not milvus.search_calls


def test_robot_prefilter_and_summary_require_explicit_canonical_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hit = _hit(tmp_path, body="must not enter tool output")
    embedding = _Embedding()
    milvus = _Milvus([hit])

    def unexpected_read(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("search must not read canonical leaf contents")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    result = _searcher(tmp_path, embedding=embedding, milvus=milvus).search_knowledge(
        query_text="  grasp a cup sideways  ",
        scope="global",
        kind="strategy",
        top_k=3,
    )
    assert not result.is_error, result.data
    assert result.images == []
    assert embedding.calls == [{"text": "grasp a cup sideways", "image_uris": []}]
    call = milvus.search_calls[0]
    assert call["collection_name"] == "leaves__v1"
    assert call["filter"] == (
        'robot == "libero" and scope == "global" and kind == "strategy"'
    )
    assert call["limit"] == 3
    assert call["anns_field"] == "vector"
    leaf = result.data["leaves"][0]
    assert leaf["leaf_uri"] == str(tmp_path / "global" / "side-grasp.md")
    assert leaf["leaf_id"] == "libero:side-grasp"
    assert leaf["distance"] == 0.91
    assert "body" not in leaf


@pytest.mark.parametrize(
    "fields",
    [
        {"robot": "robocasa"},
        {"leaf_id": "robocasa:side-grasp"},
        {"leaf_id": "libero:another-leaf"},
        {"scope": "task-family"},
        {"kind": "failure"},
        {"leaf_uri": "global/side-grasp.md"},
        {"leaf_uri": "agfs://corpus/global/side-grasp.md"},
    ],
)
def test_result_metadata_is_rechecked_after_vector_search(
    tmp_path: Path, fields: dict[str, Any]
) -> None:
    milvus = _Milvus([_hit(tmp_path, **fields)])
    result = _searcher(tmp_path, milvus=milvus).search_knowledge(
        query_text="cup grasp", scope="global", kind="strategy"
    )
    assert result.data["error"] == "knowledge_search_unavailable"
    assert "leaves" not in result.data
    assert result.data["fallback"]["index"] == str(tmp_path / "MEMORY.md")


@pytest.mark.parametrize("symlink", [False, True])
def test_search_rejects_other_robot_paths(tmp_path: Path, symlink: bool) -> None:
    root = tmp_path / "memory" / "libero"
    foreign = tmp_path / "memory" / "robocasa" / "global" / "side-grasp.md"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("another robot's memory", encoding="utf-8")
    hit = _hit(root)
    if symlink:
        path = root / "global" / "side-grasp.md"
        path.unlink()
        path.symlink_to(foreign)
    else:
        hit["entity"]["leaf_uri"] = str(foreign)
    result = _searcher(root, milvus=_Milvus([hit])).search_knowledge("grasp cup")
    assert result.data["error"] == "knowledge_search_unavailable"
    assert "escapes" in result.data["detail"]


def test_search_rejects_conflict_archive_paths(tmp_path: Path) -> None:
    conflict = tmp_path / "_internal" / "conflicts" / "side-grasp.md"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("unpublished conflict", encoding="utf-8")
    hit = _hit(tmp_path, leaf_uri=str(conflict))
    result = _searcher(tmp_path, milvus=_Milvus([hit])).search_knowledge("grasp cup")
    assert result.data["error"] == "knowledge_search_unavailable"
    assert "published" in result.data["detail"]


def test_contradicted_confidence_remains_visible(tmp_path: Path) -> None:
    hit = _hit(tmp_path, confidence="contradicted")
    result = _searcher(tmp_path, milvus=_Milvus([hit])).search_knowledge("grasp cup")
    assert not result.is_error
    assert result.data["leaves"][0]["confidence"] == "contradicted"


@pytest.mark.parametrize(
    "response",
    [
        {"model": "different-model"},
        {"dim": 12},
        {"vector": [0.0]},
        {"vector": [True] * VECTOR_DIM},
        {"vector": ["0.1"] * VECTOR_DIM},
        {"vector": [float("nan")] * VECTOR_DIM},
        {"vector": [float("inf")] * VECTOR_DIM},
    ],
)
def test_incompatible_or_invalid_embeddings_use_grep_fallback(
    tmp_path: Path, response: dict[str, Any]
) -> None:
    milvus = _Milvus()
    result = _searcher(
        tmp_path, embedding=_Embedding(**response), milvus=milvus
    ).search_knowledge("grasp cup")
    assert result.data["error"] == "knowledge_search_unavailable"
    assert not milvus.search_calls
    assert result.data["fallback"]["grep_roots"] == [
        str(tmp_path / "global"),
        str(tmp_path / "task-family"),
    ]


def test_collection_model_is_checked_before_encoding(tmp_path: Path) -> None:
    embedding = _Embedding()
    milvus = _Milvus()
    milvus.properties["embedding_model"] = "another-model"
    result = _searcher(tmp_path, embedding=embedding, milvus=milvus).search_knowledge(
        "grasp cup"
    )
    assert result.is_error
    assert not embedding.calls
    assert not milvus.search_calls


def test_service_failures_are_readonly_and_report_fallback(tmp_path: Path) -> None:
    class BrokenMilvus(_Milvus):
        def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
            raise OSError("Milvus offline")

    class BrokenEmbedding:
        def encode(self, **kwargs: Any) -> dict[str, Any]:
            raise OSError("embedding offline")

    for searcher, detail in (
        (_searcher(tmp_path, milvus=BrokenMilvus()), "Milvus offline"),
        (_searcher(tmp_path, embedding=BrokenEmbedding()), "embedding offline"),
        (
            KnowledgeSearchTool(
                robot="libero",
                memory_version=VERSION,
                embedding_client=None,
                milvus_client=_Milvus(),
                memory_root=tmp_path,
            ),
            "not configured",
        ),
    ):
        result = searcher.search_knowledge("grasp cup")
        assert result.data["error"] == "knowledge_search_unavailable"
        assert detail in result.data["detail"]
        assert "read_text_file" in result.data["fallback"]["instruction"]


def test_disabled_or_missing_index_does_not_connect_or_create_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_connect(**kwargs: Any) -> Any:
        raise AssertionError("disabled or absent local index must not connect")

    monkeypatch.setattr(
        "rpent.memory.knowledge_search.MilvusClient", unexpected_connect
    )
    toolkit = _Toolkit()
    kwargs = {
        "robot": "libero",
        "memory_version": VERSION,
        "embedding_client": _Embedding(),
        "memory_root": tmp_path / "libero",
    }
    assert (
        register_knowledge_search(
            toolkit, enabled=False, milvus_uri="http://milvus:19530", **kwargs
        )
        is None
    )
    assert create_knowledge_search(**kwargs) is None
    assert not toolkit.tools
    assert not (tmp_path / "libero").exists()


def test_registration_requires_collection_and_preserves_external_client_ownership(
    tmp_path: Path,
) -> None:
    toolkit = _Toolkit()
    milvus = _Milvus()
    kwargs = {
        "robot": "libero",
        "memory_version": VERSION,
        "embedding_client": _Embedding(),
        "memory_root": tmp_path,
        "milvus_client": milvus,
    }
    assert register_knowledge_search(toolkit, enabled=False, **kwargs) is None
    assert not milvus.collection_calls
    milvus.collection_exists = False
    assert register_knowledge_search(toolkit, enabled=True, **kwargs) is None
    assert not toolkit.tools
    milvus.collection_exists = True
    search = register_knowledge_search(toolkit, enabled=True, **kwargs)
    assert search is not None
    assert [declaration.name for declaration in toolkit.tools] == ["search_knowledge"]
    assert toolkit.tools[0].readonly is True
    search.close()
    assert milvus.closed == 0


def test_factory_owned_client_is_closed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    milvus = _Milvus()
    monkeypatch.setattr(
        "rpent.memory.knowledge_search.MilvusClient", lambda **kwargs: milvus
    )
    search = create_knowledge_search(
        robot="libero",
        memory_version=VERSION,
        embedding_client=_Embedding(),
        memory_root=tmp_path,
        milvus_uri="http://milvus:19530",
    )
    assert search is not None
    search.close()
    search.close()
    assert milvus.closed == 1


def test_unavailable_index_cleanup_failure_still_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenMilvus(_Milvus):
        def has_collection(self, **kwargs: Any) -> bool:
            raise OSError("Milvus unavailable")

        def close(self) -> None:
            raise OSError("Milvus close unavailable")

    monkeypatch.setattr(
        "rpent.memory.knowledge_search.MilvusClient", lambda **kwargs: BrokenMilvus()
    )
    assert (
        create_knowledge_search(
            robot="libero",
            memory_version=VERSION,
            embedding_client=_Embedding(),
            memory_root=tmp_path,
            milvus_uri="http://milvus:19530",
        )
        is None
    )


@pytest.mark.parametrize("released", [False, True])
def test_real_index_separates_identical_canonical_ids_by_robot(
    tmp_path: Path, released: bool
) -> None:
    from rpent.memory.knowledge import MilvusKnowledgeSink

    roots = {robot: tmp_path / "memory" / robot for robot in ("libero", "robocasa")}
    for root in roots.values():
        path = root / "global" / "side-grasp.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            "---\n"
            "id: side-grasp\n"
            "scope: global\n"
            "kind: strategy\n"
            "title: Side grasp\n"
            "applies_when: A mug handle is visible\n"
            "confidence: verified\n"
            "evidence:\n"
            "  cells: [seed0]\n"
            "---\n"
            "Approach the mug handle horizontally.\n",
            encoding="utf-8",
        )
    embedding = _Embedding()
    sink = MilvusKnowledgeSink(
        embedding,
        memory_root=roots["libero"],
        robot="libero",
        memory_version=VERSION,
    )
    try:
        sink.upsert_paths([roots["libero"] / "global" / "side-grasp.md"])
        other_sink = MilvusKnowledgeSink(
            embedding,
            memory_root=roots["robocasa"],
            robot="robocasa",
            memory_version=VERSION,
            client=sink.client,
        )
        other_sink.upsert_paths([roots["robocasa"] / "global" / "side-grasp.md"])
        assert (
            sink.client.get_collection_stats(collection_name="leaves__v1")["row_count"]
            == 2
        )
        if released:
            sink.client.release_collection(collection_name="leaves__v1")
        for robot, root in roots.items():
            result = KnowledgeSearchTool(
                robot=robot,
                memory_version=VERSION,
                embedding_client=embedding,
                milvus_client=sink.client,
                memory_root=root,
            ).search_knowledge("How should I approach a mug handle?", top_k=5)
            assert not result.is_error, result.data
            assert [leaf["leaf_id"] for leaf in result.data["leaves"]] == [
                f"{robot}:side-grasp"
            ]
            assert result.data["leaves"][0]["leaf_uri"] == str(
                root / "global" / "side-grasp.md"
            )
        toolkit = _Toolkit()
        search = register_knowledge_search(
            toolkit,
            enabled=True,
            robot="libero",
            memory_version=VERSION,
            embedding_client=embedding,
            memory_root=roots["libero"],
        )
        assert search is not None
        try:
            result = toolkit.tools[0]("How should I approach a mug handle?")
            assert not result.is_error, result.data
            assert result.data["leaves"][0]["leaf_id"] == "libero:side-grasp"
        finally:
            search.close()
    finally:
        sink.close()
