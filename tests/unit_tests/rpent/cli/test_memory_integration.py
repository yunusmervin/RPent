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

"""Evaluation remains readonly across terminal and episode memory integrations."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.cli import main as cli
from rpent.robots import PromptBundle, RobotSpec, RunConfig


class ReachedPlanner(Exception):
    pass


def _spec(root: Path):
    return RobotSpec(
        name="offline",
        prompts=PromptBundle(
            system=lambda variables: "system", user=lambda variables: "task"
        ),
        add_cli_args=lambda parser, use_dashboard: None,
        parse_config=lambda args: RunConfig(
            recipe_tag="cell",
            output_dir=root / "run",
            prompt_vars={"memory_dir": str(root / "memory" / "offline")},
            task_desc={},
        ),
        init_runtime=lambda *args: pytest.fail("no runtime should be started"),
        supports_exploration=True,
    )


@pytest.mark.parametrize("explore", [False, True])
def test_terminal_session_writes_require_exploration(tmp_path, monkeypatch, explore):
    spec = _spec(tmp_path)
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("offline",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli, "init_output_dir", lambda directory, **kwargs: directory)
    monkeypatch.setattr(
        cli,
        "create_session_memory_client",
        lambda endpoint: SimpleNamespace(add=lambda **kwargs: None),
    )
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    writable = []

    def build(*args, **kwargs):
        writable.append(kwargs["user_input_hook"].__self__.writable)
        raise ReachedPlanner

    monkeypatch.setattr(cli, "build_planner", build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "offline",
            "--tui",
            "--memory-profile",
            "local",
            *(["--explore"] if explore else []),
        ],
    )
    with pytest.raises(ReachedPlanner):
        cli.main()
    assert writable == [explore]


def test_episode_recording_flag_cannot_enable_evaluation_writes(
    tmp_path, monkeypatch, capsys
):
    spec = _spec(tmp_path)
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("offline",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "offline",
            "--record-episodes",
            "--memory-profile",
            "local",
        ],
    )
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    assert "--record-episodes requires --explore" in capsys.readouterr().err
