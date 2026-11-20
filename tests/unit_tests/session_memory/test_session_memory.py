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

import contextlib
import queue
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from rpent.robots import PromptBundle
from rpent.session_memory.hook import SessionMemoryHook
from rpent.session_memory.recall import (
    PreferenceRecallTools,
    recall_preferences,
)
from rpent.session_memory.service import (
    SESSION_MEMORY_COLLECTION,
    SessionMemoryClient,
    SessionMemoryFacade,
    SessionMemoryService,
    build_mem0_config,
)
from rpent.utils.rpc.http_rpc import HttpRpcServer


class FakeMem0:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []

    def add(self, messages, *, user_id, run_id, metadata):
        self.calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "run_id": run_id,
                "metadata": metadata,
            }
        )
        self.records.append(
            {
                "memory": messages[0]["content"],
                "user_id": user_id,
                "run_id": run_id,
                "metadata": metadata,
            }
        )
        return {"ok": True}

    def search(self, query, *, user_id, run_id, limit):
        del query
        return {
            "results": [
                {"memory": record["memory"]}
                for record in self.records
                if record["user_id"] == user_id and record["run_id"] == run_id
            ][:limit]
        }


class FakeClient:
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    def add(self, **kwargs):
        self.writes.append(kwargs)


def test_milvus_config_is_pinned_and_graph_memory_is_rejected() -> None:
    config = build_mem0_config(
        milvus_uri="http://milvus:19530",
        config={
            "llm": {"provider": "ollama", "config": {"model": "local"}},
            "vector_store": {
                "provider": "qdrant",
                "config": {"collection_name": "wrong", "token": "local-token"},
            },
        },
    )

    assert config["llm"]["provider"] == "ollama"
    assert config["vector_store"] == {
        "provider": "milvus",
        "config": {
            "collection_name": SESSION_MEMORY_COLLECTION,
            "embedding_model_dims": 1536,
            "url": "http://milvus:19530",
            "token": "local-token",
        },
    }
    with pytest.raises(ValueError, match="graph memory"):
        build_mem0_config(config={"graph": {"provider": "neo4j"}})


@pytest.mark.parametrize(
    ("args", "allowed"),
    [
        (
            SimpleNamespace(
                explore=True, tui=False, dashboard=False, memory_profile="local"
            ),
            True,
        ),
        (
            SimpleNamespace(
                explore=False, tui=True, dashboard=False, memory_profile="local"
            ),
            True,
        ),
        (
            SimpleNamespace(
                explore=False, tui=False, dashboard=False, memory_profile="local"
            ),
            False,
        ),
        (
            SimpleNamespace(
                explore=True, tui=True, dashboard=False, memory_profile="hf"
            ),
            False,
        ),
        (
            SimpleNamespace(
                explore=False, tui=True, dashboard=False, memory_profile="hf"
            ),
            False,
        ),
    ],
)
def test_write_gate_blocks_evaluation_and_hf(args, allowed) -> None:
    client = FakeClient()
    args.operator_id = "alice"
    hook = SessionMemoryHook.from_args(args, client, run_id="cell-a")

    hook.record_user_message("以后都用中文汇报")
    hook.record_user_message("继续")
    hook.record_verdict("success")
    hook.record_verdict("failure")
    hook.record_verdict("abort")

    expected = 4 if allowed else 0
    assert len(client.writes) == expected
    if allowed:
        assert [write["event"] for write in client.writes] == [
            "task",
            "user_input",
            "verdict",
            "verdict",
        ]
        assert all(write["user_id"] == "alice" for write in client.writes)
        assert all(write["run_id"] == "cell-a" for write in client.writes)


def test_first_task_is_recorded_only_once() -> None:
    client = FakeClient()
    hook = SessionMemoryHook(client, user_id="alice", run_id="cell-a", writable=True)

    hook.record_task("task")
    hook.record_task("duplicate")
    hook.record_user_message("steering")

    assert [write["content"] for write in client.writes] == ["task", "steering"]


def test_two_round_preference_recall_and_scope_isolation() -> None:
    memory = FakeMem0()
    service = SessionMemoryService(memory=memory)
    first_round = SessionMemoryHook(
        service, user_id="alice", run_id="cell-a", writable=True
    )
    first_round.record_user_message("以后都用中文汇报")
    service.add(
        content="其他操作者的偏好",
        event="user_input",
        user_id="bob",
        run_id="cell-a",
    )
    service.add(
        content="其他任务的偏好",
        event="user_input",
        user_id="alice",
        run_id="cell-b",
    )

    second_round_preferences = recall_preferences(
        service,
        "抓取任务",
        user_id="alice",
        run_id="cell-a",
    )
    bob_preferences = recall_preferences(
        service, "抓取任务", user_id="bob", run_id="cell-a"
    )

    assert "以后都用中文汇报" in second_round_preferences
    assert "其他操作者" not in second_round_preferences
    assert "其他任务" not in second_round_preferences
    assert "其他操作者的偏好" in bob_preferences


def test_prompt_injection_and_empty_value() -> None:
    bundle = PromptBundle(
        system=lambda variables: "System prompt", user=lambda variables: "Task"
    )

    empty = bundle.render("system", variables={"operator_preferences": ""})
    rendered = bundle.render(
        "system", variables={"operator_preferences": "- 以后都用中文汇报"}
    )
    user = bundle.render(
        "user", variables={"operator_preferences": "- 不应出现在用户模板"}
    )

    assert empty == "System prompt\n"
    assert "OPERATOR PREFERENCES" in rendered
    assert "以后都用中文汇报" in rendered
    assert user == "Task\n"


def test_readonly_recall_tool_uses_scope() -> None:
    memory = FakeMem0()
    service = SessionMemoryService(memory=memory, read_only=True)
    service._memory.add(
        [{"role": "user", "content": "中文"}],
        user_id="alice",
        run_id="cell-a",
        metadata={"event": "user_input"},
    )
    tools = PreferenceRecallTools(service, user_id="alice", run_id="cell-a")

    result = tools.recall_preferences("汇报", top_k=5)

    assert result.data["operator_preferences"] == "- 中文"
    with pytest.raises(PermissionError, match="read-only"):
        service.add(
            content="禁止写入",
            event="user_input",
            user_id="alice",
            run_id="cell-a",
        )


def test_service_and_recall_failures_degrade_with_warning(caplog) -> None:
    class BrokenClient:
        def add(self, **kwargs):
            raise RuntimeError("mem0 unavailable")

        def search(self, **kwargs):
            raise RuntimeError("mem0 unavailable")

    hook = SessionMemoryHook(
        BrokenClient(), user_id="alice", run_id="cell-a", writable=True
    )
    hook.record_user_message("task")
    recalled = recall_preferences(
        BrokenClient(), "task", user_id="alice", run_id="cell-a"
    )

    assert recalled == ""
    assert "session memory write skipped" in caplog.text
    assert "session memory recall unavailable" in caplog.text


def test_real_http_rpc_preserves_scope_and_readonly_contract() -> None:
    memory = FakeMem0()
    facade = SessionMemoryFacade(memory=memory, read_only=False)
    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = SessionMemoryClient("127.0.0.1:%d" % server.server_port)
    try:
        client.add(
            content="以后都用中文汇报",
            event="user_input",
            user_id="alice",
            run_id="cell-a",
        )
        response = client.search(
            query="汇报", user_id="alice", run_id="cell-a", top_k=5
        )
        isolated = client.search(query="汇报", user_id="bob", run_id="cell-a", top_k=5)
        assert response == {"results": [{"memory": "以后都用中文汇报"}]}
        assert isolated == {"results": []}
        assert memory.calls[0]["metadata"] == {"event": "user_input"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_tui_reader_observes_initial_and_followup_lines(monkeypatch) -> None:
    from rpent.cli import tui

    lines = iter(["initial task", "/success", "next instruction"])
    observed = []

    class Session:
        def __init__(self, **kwargs):
            pass

        def prompt(self, *args, **kwargs):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError from None

    monkeypatch.setattr(tui, "PromptSession", Session)
    monkeypatch.setattr(
        tui.sys, "stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 0)
    )
    monkeypatch.setattr(tui, "_restore_tty_on_exit", lambda fd: None)
    monkeypatch.setattr(tui, "patch_stdout", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        tui, "_route_console_logs_to_current_stdout", contextlib.nullcontext
    )
    messages = queue.Queue()
    thread = tui.start_interactive_reader(
        messages,
        line_handler=lambda line: line == "/success",
        on_user_line=observed.append,
    )
    thread.join(2)

    assert not thread.is_alive()
    assert observed == ["initial task", "next instruction"]
    assert messages.get_nowait() == "initial task"
    assert messages.get_nowait() == "next instruction"
