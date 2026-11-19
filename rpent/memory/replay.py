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

"""Run one evaluation twice against a checked, frozen memory corpus."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# These are clock metadata emitted by the CLI, planners, and tool results.
# Keep counters, IDs, paths, and message strings so real differences remain visible.
CLOCK_FIELDS = frozenset(
    {
        "timestamp",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "frozen_at",
        "elapsed_s",
        "agent_elapsed_s",
        "duration_s",
        "duration_ms",
    }
)
_RESERVED_OPTIONS = frozenset(
    {
        "--robot",
        "--env",
        "--suite",
        "--task",
        "--seed",
        "--output-dir",
        "--memory-profile",
        "--memory-dir",
        "--memory-repo",
        "--memory-version",
        "--memory-backend",
        "--agfs-endpoint",
        "--milvus-uri",
        "--explore",
        "--record-episodes",
        "--dashboard",
        "--help",
        "-h",
    }
)


def normalize_transcript(value: Any) -> Any:
    """Remove only named clock fields, recursively, from parsed JSON."""
    if isinstance(value, dict):
        normalized = {
            key: normalize_transcript(item)
            for key, item in value.items()
            if key not in CLOCK_FIELDS
        }
        # The API planner stores ToolResult.to_text() as a JSON string.
        # Decode only tool messages: ordinary model text must remain verbatim.
        if value.get("role") == "tool" and isinstance(value.get("content"), str):
            try:
                payload = json.loads(value["content"])
            except ValueError:
                pass
            else:
                if isinstance(payload, (dict, list)):
                    normalized["content"] = normalize_transcript(payload)
        return normalized
    if isinstance(value, list):
        return [normalize_transcript(item) for item in value]
    return value


def read_transcript(path: Path) -> list[dict[str, Any]]:
    """Read the JSON records appended by the CLI, including multi-record files."""
    contents = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records = []
    offset = 0
    while offset < len(contents):
        if contents[offset].isspace():
            offset += 1
            continue
        record, offset = decoder.raw_decode(contents, offset)
        if not isinstance(record, dict):
            raise ValueError(f"transcript must contain JSON objects: {path}")
        records.append(record)
    if not records:
        raise ValueError(f"transcript is empty: {path}")
    return records


def first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    """Return the first differing JSON location, preserving list order and types."""
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            location = f"{path}[{json.dumps(key, ensure_ascii=False)}]"
            if key not in left or key not in right:
                return location
            if difference := first_difference(left[key], right[key], location):
                return difference
        return None
    if isinstance(left, list):
        for index, (first, second) in enumerate(zip(left, right, strict=False)):
            if difference := first_difference(first, second, f"{path}[{index}]"):
                return difference
        return (
            f"{path}[{min(len(left), len(right))}]" if len(left) != len(right) else None
        )
    return path if left != right else None


def compare_transcripts(first: Path, second: Path, output_dir: Path) -> str | None:
    """Save normalized transcripts and return their first divergence, if any."""
    snapshots = []
    for number, directory in enumerate((first, second), start=1):
        transcripts = sorted(directory.glob("transcript_*.json"))
        if not transcripts:
            raise ValueError(f"no transcript_*.json produced in {directory}")
        snapshot = {
            transcript.name: normalize_transcript(read_transcript(transcript))
            for transcript in transcripts
        }
        (output_dir / f"run-{number}.normalized.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        snapshots.append(snapshot)
    return first_difference(*snapshots)


def _validate_extra_args(arguments: Sequence[str]) -> None:
    for argument in arguments:
        if argument.split("=", maxsplit=1)[0] in _RESERVED_OPTIONS:
            raise ValueError(
                f"evaluation option is controlled by this script: {argument}"
            )
        # argparse accepts long-option abbreviations, which could bypass the guard.
        if argument.startswith("--") and any(
            option.startswith(argument.split("=", maxsplit=1)[0])
            for option in _RESERVED_OPTIONS
        ):
            raise ValueError(f"ambiguous evaluation option: {argument}")


def _load_corpus(memory_root: Path, version: str) -> tuple[Path, Path, bytes, str]:
    manifest_path = memory_root / "_internal" / "freeze" / f"{version}.manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    corpus_root = Path(manifest["corpus_root"])
    if not corpus_root.is_absolute():
        raise ValueError("freeze manifest corpus_root must be an absolute path")
    corpus_root = corpus_root.resolve(strict=True)
    if not corpus_root.is_relative_to(memory_root):
        raise ValueError(
            "freeze manifest corpus_root escapes the robot memory directory"
        )
    return corpus_root, manifest_path, manifest_bytes, manifest["milvus_uri"]


def replay_consistency(
    args: argparse.Namespace,
    *,
    execute: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str | None:
    """Check the freeze around each run, then compare the two task transcripts."""
    for value in (args.robot, args.version):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("robot and version must each be a single directory name")
    extra_args = args.runner_args
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    _validate_extra_args(extra_args)
    memory_parent = args.memory_dir.expanduser().resolve()
    memory_root = (memory_parent / args.robot).resolve()
    if not memory_root.is_relative_to(memory_parent):
        raise ValueError("robot memory directory escapes --memory-dir")
    check_command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "tools" / "check_freeze.py"),
        "--robot",
        args.robot,
        "--version",
        args.version,
        "--memory-dir",
        str(memory_parent),
    ]
    if args.milvus_uri:
        check_command.extend(["--milvus-uri", args.milvus_uri])
    execute(check_command, check=True)
    corpus_root, manifest_path, manifest_bytes, frozen_milvus_uri = _load_corpus(
        memory_root, args.version
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(memory_root):
        raise ValueError("replay output must be outside the robot memory directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    # Both invocations receive the same path: SDK transcript stats include paths.
    working_dir = output_dir / "work"
    command = [
        args.runner,
        "--robot",
        args.robot,
        "--suite",
        args.suite,
        "--task",
        str(args.task),
        "--seed",
        str(args.seed),
        "--memory-profile",
        "local",
        "--memory-dir",
        str(corpus_root),
        "--output-dir",
        str(working_dir),
    ]
    command.extend(["--milvus-uri", args.milvus_uri or frozen_milvus_uri])
    command.extend(extra_args)
    for number in (1, 2):
        if number == 2:
            execute(check_command, check=True)
        if manifest_path.read_bytes() != manifest_bytes:
            raise RuntimeError("freeze manifest changed between replay runs")
        print(
            f"Replay {number}/2: {args.suite} task={args.task} seed={args.seed}",
            flush=True,
        )
        try:
            execute(command, check=True)
        finally:
            execute(check_command, check=True)
            if manifest_path.read_bytes() != manifest_bytes:
                raise RuntimeError("freeze manifest changed during replay")
        working_dir.rename(output_dir / f"run-{number}")
    return compare_transcripts(output_dir / "run-1", output_dir / "run-2", output_dir)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--memory-dir", type=Path, default=Path("memory"))
    parser.add_argument("--milvus-uri")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--runner",
        default="rpent",
        help="Task executable (default: rpent); no shell expansion.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Additional task arguments after --.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run two evaluations and report whether their normalized transcripts match."""
    args = _build_argparser().parse_args(argv)
    try:
        difference = replay_consistency(args)
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    if difference is not None:
        print(f"Replay mismatch: first divergence at {difference}", file=sys.stderr)
        print(f"Compare {args.output_dir}/run-{{1,2}}.normalized.json", file=sys.stderr)
        return 1
    print("Replay consistent: normalized transcript diff is empty.")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
