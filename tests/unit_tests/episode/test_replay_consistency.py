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

"""Replay controls and transcript comparison without simulator dependencies."""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from rpent.memory.replay import (
    _validate_extra_args,
    compare_transcripts,
    first_difference,
    normalize_transcript,
    read_transcript,
    replay_consistency,
)


def _arguments(tmp_path: Path) -> Namespace:
    memory_root = tmp_path / "memory" / "libero"
    corpus_root = memory_root / "v1"
    corpus_root.mkdir(parents=True)
    manifest = memory_root / "_internal" / "freeze" / "v1.manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"corpus_root": str(corpus_root), "milvus_uri": "frozen.db"})
    )
    return Namespace(
        robot="libero",
        version="v1",
        suite="libero_10_task",
        task=2,
        seed=0,
        memory_dir=tmp_path / "memory",
        milvus_uri=None,
        output_dir=tmp_path / "results",
        runner="offline-runner",
        runner_args=["--", "--planner", "api"],
    )


def test_compare_removes_clock_fields_but_keeps_behavior(tmp_path: Path) -> None:
    for number in (1, 2):
        output = tmp_path / f"run-{number}"
        output.mkdir()
        (output / "transcript_task.json").write_text(
            json.dumps(
                {
                    "elapsed_s": number,
                    "stats": {"duration_ms": number, "tool_calls": 3},
                    "messages": [{"timestamp": number, "content": "same action"}],
                }
            )
        )
    first, second = tmp_path / "run-1", tmp_path / "run-2"
    assert compare_transcripts(first, second, tmp_path) is None
    record = read_transcript(second / "transcript_task.json")[0]
    record["messages"][0]["content"] = "different action"
    (second / "transcript_task.json").write_text(json.dumps(record))
    assert compare_transcripts(first, second, tmp_path) == (
        '$["transcript_task.json"][0]["messages"][0]["content"]'
    )


def test_read_transcript_accepts_appended_json_and_rejects_empty(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_text('{"first": true}\n{\n"second": false\n}')
    assert read_transcript(transcript) == [{"first": True}, {"second": False}]
    transcript.write_text(" \n")
    with pytest.raises(ValueError, match="empty"):
        read_transcript(transcript)


def test_normalize_decodes_tool_json_without_altering_model_text() -> None:
    original = [
        {"role": "tool", "content": '{"agent_elapsed_s": 1, "action": "grasp"}'},
        {"role": "assistant", "content": '{"timestamp": 1}'},
        {"role": "tool", "content": "elapsed_s=1 [truncated]"},
    ]
    normalized = normalize_transcript(original)
    assert normalized == [
        {"role": "tool", "content": {"action": "grasp"}},
        {"role": "assistant", "content": '{"timestamp": 1}'},
        {"role": "tool", "content": "elapsed_s=1 [truncated]"},
    ]
    changed = normalize_transcript(
        [{"role": "tool", "content": '{"agent_elapsed_s": 2, "action": "place"}'}]
    )
    assert first_difference(normalized[:1], changed) == '$[0]["content"]["action"]'
    assert isinstance(original[0]["content"], str)


@pytest.mark.parametrize(
    "arguments", [["--explore"], ["--record-episodes"], ["--memory-p=hf"], ["--seed=7"]]
)
def test_replay_rejects_overrides_and_cli_abbreviations(arguments: list[str]) -> None:
    with pytest.raises(ValueError, match="option"):
        _validate_extra_args(arguments)


def test_replay_runs_same_real_executable_twice_and_checks_each_freeze(
    tmp_path: Path,
) -> None:
    args = _arguments(tmp_path)
    runner = tmp_path / "offline-runner"
    runner.write_text(
        f"#!{sys.executable}\n"
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "for flag in ('robot', 'suite', 'task', 'seed', 'memory-profile', 'memory-dir', 'output-dir', 'milvus-uri', 'planner'):\n"
        "    parser.add_argument('--' + flag)\n"
        "args = parser.parse_args()\n"
        "out = Path(args.output_dir)\n"
        "out.mkdir()\n"
        "(out / 'transcript_task.json').write_text(json.dumps(vars(args)))\n"
    )
    runner.chmod(0o755)
    args.runner = str(runner)
    checks = []
    invocations = []

    def execute(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        assert check
        if command[1].endswith("check_freeze.py"):
            checks.append(command)
            return subprocess.CompletedProcess(command, 0)
        invocations.append(command)
        return subprocess.run(command, check=True)

    assert replay_consistency(args, execute=execute) is None
    assert len(checks) == 4
    assert len(invocations) == 2
    assert invocations[0] == invocations[1]
    record = read_transcript(args.output_dir / "run-1" / "transcript_task.json")[0]
    assert record["memory_profile"] == "local"
    assert record["memory_dir"] == str(args.memory_dir / "libero" / "v1")
    assert record["milvus_uri"] == "frozen.db"
    assert not (args.output_dir / "work").exists()


def test_replay_stops_before_runner_when_freeze_check_fails(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    calls = []

    def execute(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        replay_consistency(args, execute=execute)
    assert len(calls) == 1
    assert not args.output_dir.exists()


def test_replay_detects_manifest_replacement_during_first_run(tmp_path: Path) -> None:
    args = _arguments(tmp_path)

    def execute(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        if not command[1].endswith("check_freeze.py"):
            manifest = (
                args.memory_dir / "libero" / "_internal" / "freeze" / "v1.manifest.json"
            )
            manifest.write_text(manifest.read_text() + "\n")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(RuntimeError, match="manifest changed during replay"):
        replay_consistency(args, execute=execute)


def test_replay_checks_freeze_after_runner_failure(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    checks = []

    def execute(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        if command[1].endswith("check_freeze.py"):
            checks.append(command)
            return subprocess.CompletedProcess(command, 0)
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        replay_consistency(args, execute=execute)
    assert len(checks) == 2


def test_first_difference_keeps_types_order_and_presence() -> None:
    assert first_difference({"done": False}, {"done": 0}) == '$["done"]'
    assert first_difference([1, 2], [2, 1]) == "$[0]"
    assert first_difference({"value": None}, {}) == '$["value"]'
