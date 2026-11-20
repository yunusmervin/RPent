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

"""Memory retrieval coexists with the latest LIBERO perception and capture."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robots.libero import toolkit as libero_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import FileSystemBackend
from rpent.session.env_capture import CapturingEnvClient
from rpent.utils.config import EMBEDDING_MODEL_ID


@pytest.mark.parametrize("mode", ["evaluation", "exploration"])
def test_memory_search_and_geometry_keep_capture_and_drain_before_cleanup(
    tmp_path, monkeypatch, fake_single_arm_primitives, mode
):
    primitives = fake_single_arm_primitives.for_robot(
        libero_toolkit.libero_tools.LiberoPrimitives
    )
    monkeypatch.setattr(libero_toolkit.libero_tools, "LiberoPrimitives", primitives)
    monkeypatch.setattr(
        libero_toolkit.libero_tools,
        "dump_state",
        lambda *args, **kwargs: SimpleNamespace(step_idx=0, terminated=False),
    )
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    search_started, release_search = threading.Event(), threading.Event()

    def search(**kwargs):
        search_started.set()
        assert release_search.wait(timeout=5)
        return [[]]

    episode_index = Mock(has_collection=Mock(return_value=True))
    knowledge_index = Mock(
        has_collection=Mock(return_value=True),
        describe_collection=Mock(
            return_value={"properties": {"embedding_model": EMBEDDING_MODEL_ID}}
        ),
        search=Mock(side_effect=search),
    )
    monkeypatch.setattr(
        "rpent.episode.search_tool.MilvusClient", lambda **kwargs: episode_index
    )
    monkeypatch.setattr(
        "rpent.memory.knowledge_search.MilvusClient", lambda **kwargs: knowledge_index
    )
    embedding = Mock(
        encode=Mock(
            return_value={
                "vector": [1.0] * 1536,
                "dim": 1536,
                "model": EMBEDDING_MODEL_ID,
            }
        )
    )
    env, sam3 = object(), object()
    runtime_kwargs = {"env": env, "sam3_client": sam3, "embedding_client": embedding}
    toolkit = libero_toolkit.LiberoToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=NullDashboardEventSink(),
        memory=FileSystemBackend(
            tmp_path / "memory" / "libero",
            memory_access="inbox_write" if mode == "exploration" else "read_only",
            inbox_cell_tag="cell",
        ),
        state_output_dir=tmp_path / "state",
        mode=mode,
        geometry_grounder=object(),
        episode_search_enabled=True,
        knowledge_search_enabled=True,
        memory_version="v1",
        milvus_uri="http://offline-index",
    )
    declarations = {definition.name: definition for definition in toolkit.list_tools()}
    for name in (
        "geometry",
        "segment",
        "back_project",
        "search_episodes",
        "search_knowledge",
    ):
        assert declarations[name].readonly
    primitive_kwargs = primitives.instances[-1].kwargs
    captured_env = primitive_kwargs["env"]
    assert isinstance(captured_env, CapturingEnvClient) is (mode == "evaluation")
    assert (captured_env.wrapped if mode == "evaluation" else captured_env) is env
    assert primitive_kwargs["sam3_client"] is sam3
    assert "embedding_client" not in primitive_kwargs
    assert runtime_kwargs["embedding_client"] is embedding

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = pool.submit(
            toolkit.execute_tool, "search_knowledge", {"query_text": "grasp a bowl"}
        )
        try:
            assert search_started.wait(timeout=5)
            closing = pool.submit(toolkit.close)
            with toolkit._scheduler.condition:
                assert toolkit._scheduler.condition.wait_for(
                    lambda: toolkit._scheduler.closed, timeout=5
                )
            episode_index.close.assert_not_called()
            knowledge_index.close.assert_not_called()
        finally:
            release_search.set()
        closing.result(timeout=5)
        assert result.result(timeout=5).data == {"leaves": []}

    episode_index.close.assert_called_once_with()
    knowledge_index.close.assert_called_once_with()
    embedding.close.assert_not_called()
