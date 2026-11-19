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

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rpent.cli.main import _build_argparser
from rpent.episode.hook import record_episode
from rpent.episode.writer import EpisodeWriter
from rpent.memory import AGFSBackend
from rpent.utils.config import EMBEDDING_MODEL_ID

CREATED = datetime(2026, 11, 10, 3, 15, tzinfo=UTC)


def _run_directory(root: Path, *, truncated: bool = False) -> Path:
    run = root / "run"
    records = []
    for step, content in enumerate((b"first", b"middle", b"last")):
        artifacts = ["agentview.png"]
        if step == 2:
            artifacts.extend(["segment_overlay_00.png", "segment_overlay_01.png"])
        record = {
            "step_idx": step,
            "artifacts": artifacts,
            "terminated": step == 2 and not truncated,
            "truncated": step == 2 and truncated,
            "extras": {"task_language": "pick up the black bowl"},
        }
        records.append(record)
        frame = run / "agentview.png" / f"{step:02d}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(content)
    for overlay, content in (
        ("segment_overlay_00.png", b"old-overlay"),
        ("segment_overlay_01.png", b"latest-overlay"),
    ):
        path = run / overlay / "02.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    run.mkdir(parents=True, exist_ok=True)
    (run / "states.json").write_text(json.dumps({"steps": records}))
    return run


def _backend(root: Path, *, read_only: bool = False) -> AGFSBackend:
    return AGFSBackend(
        root / "memory" / "libero",
        agfs_endpoint="http://127.0.0.1:18080",
        robot="libero",
        memory_access="read_only" if read_only else "inbox_write",
        inbox_cell_tag=None if read_only else "libero10_swap_t2_s0",
    )


def _write(run: Path, backend: AGFSBackend, outcome: str = "success"):
    return EpisodeWriter(backend).write_from_run(
        run_state_dir=run,
        cell_tag="libero10_swap_t2_s0",
        robot="libero",
        task_desc={"suite": "libero_10_swap", "task": 2, "seed": 0},
        outcome=outcome,
        stats={"tool_calls": 3, "total_output_tokens": 12},
        elapsed_s=4.5,
        created_at=CREATED,
    )


def test_writer_selects_key_frames_and_is_idempotent(tmp_path: Path) -> None:
    run = _run_directory(tmp_path)
    backend = _backend(tmp_path)

    first_episode, first_uri = _write(run, backend)
    second_episode, second_uri = _write(run, backend)

    assert first_episode == second_episode
    assert first_uri == second_uri
    assert first_episode.episode_id == "libero10_swap_t2_s0_20261110T031500"
    assert first_episode.memory_ids == ["task-family_libero10_swap_t2"]
    assert first_episode.memory_version == "unversioned"
    assert first_episode.embedding_model == EMBEDDING_MODEL_ID
    assert first_episode.assets.failure_frames == []
    assert len(first_episode.assets.seg_overlays) == 1

    expected = {}
    for content in (b"first", b"last", b"latest-overlay"):
        digest = hashlib.sha256(content).hexdigest()
        expected[content] = (
            f"agfs://frames/sha256/{digest[:2]}/{digest[2:4]}/{digest}.png"
        )
    assert first_episode.assets.first_frame == expected[b"first"]
    assert first_episode.assets.success_frame == expected[b"last"]
    assert first_episode.assets.seg_overlays == [expected[b"latest-overlay"]]

    episode_path = backend.episodes_dir / "unversioned" / f"{first_episode.episode_id}.json"
    saved = json.loads(episode_path.read_text())
    assert saved["outcome"] == "success"
    assert saved["stats"] == {
        "elapsed_s": 4.5,
        "tool_calls": 3,
        "output_tokens": 12,
    }
    assert "base64" not in episode_path.read_text()
    assert len(list(backend.episodes_dir.rglob("*.json"))) == 1
    assert len(list(backend.frames_dir.rglob("*.png"))) == 3


def test_writer_uses_truncated_last_frame_for_failure(tmp_path: Path) -> None:
    backend = _backend(tmp_path)

    episode, _ = _write(_run_directory(tmp_path, truncated=True), backend, "failure")

    assert episode.outcome == "failure"
    assert episode.confidence == "single-shot"
    assert episode.assets.success_frame is None
    assert len(episode.assets.failure_frames) == 1
    assert episode.assets.failure_frames[0] != episode.assets.first_frame


def test_hook_degrades_to_warning_when_agfs_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenStore:
        def put_frame(self, png_bytes: bytes) -> str:
            raise OSError("AGFS unavailable")

    run = _run_directory(tmp_path)
    with caplog.at_level(logging.WARNING, logger="rpent.episode"):
        result = record_episode(
            BrokenStore(),
            run_state_dir=run,
            cell_tag="libero10_swap_t2_s0",
            robot="libero",
            task_desc={"suite": "libero_10_swap", "task": 2, "seed": 0},
            finish_result={"status": "success"},
            stats={},
            elapsed_s=1,
            created_at=CREATED,
        )

    assert result is None
    assert "run result is unchanged" in caplog.text
    assert "AGFS unavailable" in caplog.text


def test_hook_is_idempotent_for_the_same_run(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    kwargs = {
        "run_state_dir": _run_directory(tmp_path),
        "cell_tag": "libero10_swap_t2_s0",
        "robot": "libero",
        "task_desc": {"suite": "libero_10_swap", "task": 2, "seed": 0},
        "finish_result": {"status": "success"},
        "stats": {},
        "elapsed_s": 1,
        "created_at": CREATED,
    }

    first = record_episode(backend, **kwargs)
    second = record_episode(backend, **kwargs)

    assert first == second
    assert len(list(backend.episodes_dir.rglob("*.json"))) == 1


def test_explicit_recording_does_not_relax_corpus_permissions(tmp_path: Path) -> None:
    backend = _backend(tmp_path, read_only=True)
    with pytest.raises(PermissionError):
        backend.put_frame(b"frame")

    backend.enable_episode_recording()

    assert backend.put_frame(b"frame").startswith("agfs://frames/sha256/")
    assert backend.put_episode(
        memory_version="unversioned", episode_id="episode", payload=b"{}"
    ).startswith("agfs://episodes/unversioned/")
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        backend.authorize_write(backend.root / "global" / "new.md")


def test_episode_cli_options_default_off() -> None:
    parser = _build_argparser()
    defaults = parser.parse_args([])
    configured = parser.parse_args(
        ["--record-episodes", "--episode-store-endpoint", "http://localhost:9000"]
    )

    assert defaults.record_episodes is False
    assert defaults.episode_store_endpoint is None
    assert configured.record_episodes is True
    assert configured.episode_store_endpoint == "http://localhost:9000"


def test_explicit_recording_mounts_episodes_for_owned_server(tmp_path: Path) -> None:
    backend = AGFSBackend(
        tmp_path / "memory" / "libero",
        agfs_endpoint="http://127.0.0.1:18080",
        robot="libero",
        record_episodes=True,
    )

    mounts = {mount["path"] for mount in backend._mounts()}

    assert "/episodes" in mounts
    assert backend._episode_recording is True
