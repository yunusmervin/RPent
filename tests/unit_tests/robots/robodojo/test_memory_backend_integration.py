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

"""RoboDojo backend selection and exploration permissions work together."""

from types import SimpleNamespace

import pytest

from robots.robodojo import robot_spec, toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.agfs_backend import AGFSBackend
from rpent.robots.robot_spec import RunConfig


@pytest.mark.parametrize("backend_name", ["filesystem", "agfs"])
@pytest.mark.parametrize("mode", ["evaluation", "exploration"])
def test_toolkit_backend_preserves_mode_permissions(
    monkeypatch, tmp_path, backend_name, mode
):
    monkeypatch.setattr(toolkit, "RoboDojoToolkit", SimpleNamespace)
    root = tmp_path / "memory" / "robodojo"
    config = RunConfig(
        recipe_tag="stack_blocks_s0_l0",
        output_dir=tmp_path / "output",
        prompt_vars={"memory_dir": str(root)},
        task_desc={"task": "stack_blocks"},
        memory_backend=backend_name,
        agfs_endpoint="http://127.0.0.1:18080",
    )
    kit = robot_spec.get_toolkit(
        runtime_kwargs={"vla_backend": "awomo05"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode=mode,
        attempts_per_session=3,
    )

    assert kit.mode == mode
    assert kit.attempts_per_session == 3
    assert kit.primitives_kwargs == {"vla_backend": "awomo05"}
    assert kit.memory.root == root
    if backend_name == "agfs":
        assert isinstance(kit.memory, AGFSBackend)
        assert kit.memory.endpoint == config.agfs_endpoint
        assert kit.memory.daemon is None
    inbox = root / "_internal" / "inbox" / config.recipe_tag / "leaf.md"
    if mode == "exploration":
        assert kit.memory.authorize_write(inbox) == inbox
    else:
        with pytest.raises(PermissionError, match="writing to memory is denied"):
            kit.memory.authorize_write(inbox)
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        kit.memory.authorize_write(root / "global" / "leaf.md")
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        kit.memory.authorize_write(root / "_internal" / "inbox" / "other" / "leaf.md")
    assert not root.exists()
