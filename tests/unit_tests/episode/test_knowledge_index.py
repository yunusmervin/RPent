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

"""Canonical merge, optional knowledge indexing, and rebuild contracts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from robots.libero.robot_spec import _embedding_requested, get_toolkit
from rpent.cli.main import _build_argparser
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.knowledge import KnowledgeIndexHook, MilvusKnowledgeSink
from rpent.memory.knowledge_search import KnowledgeSearchTool
from rpent.memory.manager import FileSystemBackend, _split_frontmatter
from rpent.robots import RunConfig
from rpent.utils.config import EMBEDDING_MODEL_ID
from tools.rebuild_episode_index import LocalMockEmbedding
from tools.rebuild_episode_index import main as rebuild_main

VERSION = "knowledge-v1"
VECTOR_DIM = 1536


def _draft(
    root: Path, cell: str, identifier: str, body: str, *, scope: str = "global"
) -> Path:
    metadata = {
        "id": identifier,
        "scope": scope,
        "kind": "strategy",
        "title": identifier,
        "applies_when": "handling a kitchen object",
        "confidence": "single-shot",
        "evidence": {"cells": [cell], "attempts": 1},
    }
    if scope == "task-family":
        metadata.update(
            suite="libero10", regime="swap", task_id=2, task_language="lift the bowl"
        )
    path = root / "_internal" / "inbox" / cell / f"{identifier}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n" + yaml.safe_dump(metadata) + "---\n\n" + body + "\n", encoding="utf-8"
    )
    return path


def _merge(root: Path, cell: str, indexer=None, *, writable: bool = True):
    return FileSystemBackend(
        root,
        memory_access="inbox_write" if writable else "read_only",
        inbox_cell_tag=cell,
        knowledge_indexer=indexer,
    ).merge_memory(cell_tag=cell, run_state_dir=root.parent / "run", solved=False)


def test_merge_indexes_published_revision_and_never_conflict_draft(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory" / "libero"
    indexed = []

    class Indexer:
        def upsert_paths(self, paths):
            indexed.append([(path, _split_frontmatter(path)) for path in paths])
            assert (root / "MEMORY.md").is_file()
            return len(paths)

    _draft(root, "first", "bowl", "Use the rim.")
    assert _merge(root, "first", Indexer())["global"] == 1
    _draft(root, "second", "bowl", "Push hard on the center.")
    result = _merge(root, "second", Indexer())
    assert result["evidence"] == 1 and result["conflicts"] == 1
    assert len(indexed) == 2
    path, (metadata, body) = indexed[-1][0]
    assert path == root / "global" / "bowl.md"
    assert metadata["confidence"] == "probable"
    assert body.strip() == "Use the rim."
    assert (
        "Push hard on the center."
        in (root / "_internal" / "conflicts" / "bowl__from_second.md").read_text()
    )
    # An idempotent repeat must not create another index update.
    _merge(root, "second", Indexer())
    assert len(indexed) == 2


def test_unmergeable_leaf_conflict_is_not_indexed(tmp_path: Path) -> None:
    root = tmp_path / "memory" / "libero"
    published = root / "global" / "bowl.md"
    published.parent.mkdir(parents=True)
    published.write_text("Hand maintained note without metadata.")
    _draft(root, "cell", "bowl", "Incoming draft.")
    calls = []
    result = _merge(
        root, "cell", SimpleNamespace(upsert_paths=lambda paths: calls.append(paths))
    )
    assert result["conflicts"] == 1
    assert calls == []
    assert published.read_text() == "Hand maintained note without metadata."


def test_task_family_and_global_publication_share_the_index_hook(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory" / "libero"
    _draft(root, "cell", "bowl", "Use the rim.")
    _draft(
        root, "cell", "family-draft", "Lift from the visible side.", scope="task-family"
    )
    calls = []
    result = _merge(
        root, "cell", SimpleNamespace(upsert_paths=lambda paths: calls.append(paths))
    )
    assert result["global"] == result["task-family"] == 1
    assert len(calls) == 1
    assert set(calls[0]) == {
        root / "global" / "bowl.md",
        root / "task-family" / "task-family_libero10_swap_t2.md",
    }


def test_milvus_outage_preserves_complete_canonical_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "memory" / "libero"
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("Milvus unavailable for knowledge indexing")

    monkeypatch.setattr("rpent.memory.knowledge.MilvusClient", unavailable)
    hook = KnowledgeIndexHook(
        LocalMockEmbedding(root),
        memory_root=root,
        robot="libero",
        memory_version=VERSION,
    )
    assert calls == []
    _draft(root, "cell", "bowl", "Use the rim.")
    with caplog.at_level(logging.WARNING):
        result = _merge(root, "cell", hook)
    assert result["global"] == 1
    assert (root / "global" / "bowl.md").is_file()
    assert (root / "MEMORY.md").is_file()
    assert (root / "_internal" / "merged" / "cell").is_dir()
    assert calls and "canonical merge is preserved" in caplog.text
    assert "Milvus unavailable" in caplog.text


def test_readonly_backend_does_not_invoke_derived_index(tmp_path: Path) -> None:
    root = tmp_path / "memory" / "libero"
    _draft(root, "cell", "bowl", "Use the rim.")
    calls = []
    _merge(
        root,
        "cell",
        SimpleNamespace(upsert_paths=lambda paths: calls.append(paths)),
        writable=False,
    )
    assert calls == []


class ControlledSemanticEncoder:
    """A test double for synonym groups; this is not a model quality benchmark."""

    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]:
        words = set(text.lower().replace("?", "").split())
        vector = [0.0] * VECTOR_DIM
        for index, group in enumerate(
            (
                {"bowl", "basin", "ceramic"},
                {"mug", "drinking", "handle"},
                {"plate", "dish", "edge"},
            )
        ):
            if words & group:
                vector[index] = 1.0
        if not any(vector):
            vector[3] = 1.0
        return {"vector": vector, "model": EMBEDDING_MODEL_ID, "dim": VECTOR_DIM}


def test_three_paraphrased_global_leaves_recall_corresponding_leaf(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory" / "libero"
    root.mkdir(parents=True)
    encoder = ControlledSemanticEncoder()
    sink = MilvusKnowledgeSink(
        encoder, memory_root=root, robot="libero", memory_version=VERSION
    )
    try:
        examples = [
            (
                "grasp_basin",
                "Support the ceramic basin from its sides.",
                "How should I pick up a bowl?",
            ),
            (
                "grasp_vessel",
                "Lift a drinking vessel by the handle.",
                "How do I grasp a mug?",
            ),
            (
                "grasp_flatware",
                "Use an edge approach for a flat dish.",
                "How should the robot take the plate?",
            ),
        ]
        for identifier, body, _ in examples:
            _draft(root, identifier, identifier, body)
            _merge(root, identifier, sink)
        search = KnowledgeSearchTool(
            robot="libero",
            memory_version=VERSION,
            embedding_client=encoder,
            milvus_client=sink.client,
            memory_root=root,
        )
        for identifier, _, query in examples:
            result = search.search_knowledge(query_text=query, top_k=1)
            assert result.data["leaves"][0]["leaf_uri"] == str(
                root / "global" / f"{identifier}.md"
            )
            assert result.data["leaves"][0]["confidence"] == "single-shot"
    finally:
        sink.close()


def test_rebuild_tool_preserves_query_results_and_contradicted_leaf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "memory" / "libero"
    for identifier in ("bowl", "mug", "plate"):
        _draft(root, identifier, identifier, f"Hold the {identifier} securely.")
        _merge(root, identifier)
    leaf = root / "global" / "bowl.md"
    metadata, body = _split_frontmatter(leaf)
    metadata["confidence"] = "contradicted"
    leaf.write_text("---\n" + yaml.safe_dump(metadata) + "---\n" + body)
    encoder = LocalMockEmbedding(root)
    sink = MilvusKnowledgeSink(
        encoder, memory_root=root, robot="libero", memory_version=VERSION
    )
    try:
        sink.rebuild()
        search = KnowledgeSearchTool(
            robot="libero",
            memory_version=VERSION,
            embedding_client=encoder,
            milvus_client=sink.client,
            memory_root=root,
        )
        before = search.search_knowledge(
            query_text="How should I hold a bowl?", top_k=3
        ).data
        rebuild_main(
            [
                "--robot",
                "libero",
                "--memory-version",
                VERSION,
                "--memory-dir",
                str(tmp_path / "memory"),
                "--collection",
                "leaves",
                "--mock",
            ]
        )
        after = search.search_knowledge(
            query_text="How should I hold a bowl?", top_k=3
        ).data
        assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
        assert any(item["confidence"] == "contradicted" for item in after["leaves"])
        assert "Rebuilt 3 leaves" in capsys.readouterr().out
    finally:
        sink.close()


def test_knowledge_search_cli_is_opt_in_and_reaches_robot_toolkit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = _build_argparser()
    assert parser.parse_args([]).enable_knowledge_search is False
    assert (
        parser.parse_args(["--enable-knowledge-search"]).enable_knowledge_search is True
    )
    assert _embedding_requested(SimpleNamespace(enable_knowledge_search=True))
    assert not _embedding_requested(SimpleNamespace())
    config = RunConfig(
        recipe_tag="cell",
        output_dir=tmp_path / "run",
        prompt_vars={
            "memory_dir": str(tmp_path / "memory" / "libero"),
            "memory_version": VERSION,
        },
        task_desc={},
        knowledge_search_enabled=True,
    )
    monkeypatch.setattr("robots.libero.toolkit.LiberoToolkit", lambda **kwargs: kwargs)
    result = get_toolkit(
        runtime_kwargs={}, dashboard_events=NullDashboardEventSink(), config=config
    )
    assert result["knowledge_search_enabled"] is True
    assert result["memory_version"] == VERSION


def test_toolkit_attaches_lazy_writer_only_during_exploration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robots.libero.toolkit import LiberoToolkit

    monkeypatch.setattr(
        "rpent.utils.templates.default_variables",
        lambda: {"output_dir": str(tmp_path / "run")},
    )
    monkeypatch.setattr(LiberoToolkit, "init_primitives", lambda self, **kwargs: None)
    monkeypatch.setattr(LiberoToolkit, "_register_libero_tools", lambda self: None)
    for mode in ("evaluation", "exploration"):
        root = tmp_path / mode / "libero"
        memory = FileSystemBackend(
            root,
            memory_access="inbox_write" if mode == "exploration" else "read_only",
            inbox_cell_tag="cell",
        )
        toolkit = LiberoToolkit(
            runtime_kwargs={"embedding_client": ControlledSemanticEncoder()},
            dashboard_events=NullDashboardEventSink(),
            memory=memory,
            mode=mode,
            state_output_dir=tmp_path / mode / "run",
            memory_version=VERSION,
        )
        _draft(root, "cell", "bowl", "Support the ceramic basin.")
        result = memory.merge_memory(
            cell_tag="cell", run_state_dir=tmp_path / "run", solved=False
        )
        assert result["global"] == 1
        assert toolkit._knowledge_search is None
        assert (root / ".milvus").exists() is (mode == "exploration")


@pytest.mark.parametrize(
    "failure", [None, "embedding", "transcript", "interrupt", "cleanup"]
)
def test_cli_keeps_owned_embedding_alive_through_publication_and_always_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    import sys

    from rpent.cli import main as cli
    from rpent.planner.base import PlannerResult
    from rpent.robots import PromptBundle, RobotSpec

    events = []

    class Daemon:
        def __init__(self, name):
            self.name = name
            self.stopped = False

        def stop(self):
            self.stopped = True
            events.append(self.name + ":stop")
            if failure == "cleanup" and self.name == "env_server":
                raise OSError("environment daemon cleanup failure")

    embedding_daemon, environment_daemon = (
        Daemon("embedding_server"),
        Daemon("env_server"),
    )

    class Encoder(ControlledSemanticEncoder):
        def encode(self, *args, **kwargs):
            assert not embedding_daemon.stopped
            assert environment_daemon.stopped
            events.append("encode")
            if failure == "embedding":
                raise OSError("embedding service unavailable")
            return super().encode(*args, **kwargs)

    root, output = tmp_path / "memory" / "libero", tmp_path / "run"
    output.mkdir()
    encoder = Encoder()
    memory = FileSystemBackend(
        root,
        memory_access="inbox_write",
        inbox_cell_tag="cell",
        knowledge_indexer=KnowledgeIndexHook(
            encoder, memory_root=root, robot="libero", memory_version=VERSION
        ),
    )
    _draft(root, "cell", "bowl", "Support the ceramic basin.")

    class Toolkit:
        def __init__(self):
            self.memory = memory

        def solved(self):
            return False

        def close(self):
            events.append("toolkit:close")

    class Planner:
        def solve(self, **kwargs):
            if failure == "interrupt":
                raise KeyboardInterrupt
            return PlannerResult(
                finish_result={"status": "success"}, messages=[], stats={}, error=None
            )

    def add_cli_args(parser, use_dashboard):
        parser.add_argument("--auto-merge-memory", action="store_true")

    spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=add_cli_args,
        parse_config=lambda args: RunConfig(
            recipe_tag="cell",
            output_dir=output,
            prompt_vars={"memory_dir": str(root), "memory_version": VERSION},
            task_desc={},
        ),
        init_runtime=lambda *args: (
            [embedding_daemon, environment_daemon],
            {"embedding_client": encoder},
        ),
        supports_exploration=True,
    )
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli, "build_planner", lambda *args, **kwargs: Planner())
    monkeypatch.setattr(cli, "get_toolkit", lambda *args, **kwargs: Toolkit())
    monkeypatch.setattr(cli, "init_output_dir", lambda *args, **kwargs: output)
    monkeypatch.setattr(cli, "create_session_memory_client", lambda *args: None)

    def record(*args, **kwargs):
        assert not embedding_daemon.stopped
        events.append("record_episode")

    monkeypatch.setattr(cli, "record_episode", record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "libero",
            "--explore",
            "--auto-merge-memory",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(root),
            "--output-dir",
            str(output),
            "--model",
            "openai:offline",
        ],
    )
    if failure == "transcript":
        original_dump = cli.json.dump

        def dump(value, stream, **kwargs):
            if Path(stream.name).name.startswith("transcript_"):
                raise OSError("transcript disk failure")
            return original_dump(value, stream, **kwargs)

        monkeypatch.setattr(cli.json, "dump", dump)
        with pytest.raises(OSError, match="transcript disk failure"):
            cli.main()
    elif failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            cli.main()
    elif failure == "cleanup":
        with pytest.raises(OSError, match="environment daemon cleanup failure"):
            cli.main()
    else:
        assert cli.main() == 0
        assert "encode" in events and "record_episode" in events
        assert events.index("encode") < events.index("embedding_server:stop")
        assert (root / "global" / "bowl.md").is_file()
        if failure is None:
            sink = MilvusKnowledgeSink(
                encoder, memory_root=root, robot="libero", memory_version=VERSION
            )
            try:
                assert (
                    sink.client.get_collection_stats(sink.collection_name)["row_count"]
                    == 1
                )
            finally:
                sink.close()
    assert environment_daemon.stopped and embedding_daemon.stopped
