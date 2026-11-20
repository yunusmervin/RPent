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

"""Extract run artifacts and persist their canonical episode record."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from rpent.episode.model import Episode, EpisodeAssets, EpisodeStats
from rpent.utils.config import EMBEDDING_MODEL_ID

UTC = timezone.utc
DEFAULT_MEMORY_VERSION = "unversioned"
EMBEDDING_MODEL_PLACEHOLDER = EMBEDDING_MODEL_ID
_REGIMES = {"task", "swap", "lan", "object"}
_OVERLAY = re.compile(r"segment_overlay_(\d+)\.png$")


class EpisodeStore(Protocol):
    def put_frame(self, png_bytes: bytes) -> str: ...

    def put_episode(
        self,
        *,
        memory_version: str,
        episode_id: str,
        payload: bytes,
    ) -> str: ...


def _records(run_state_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_state_dir / "states.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("steps"), list):
        raise ValueError(f"invalid episode state manifest: {manifest_path}")
    records = manifest["steps"]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("episode requires at least one state record")
    return records


def _frame_path(run_state_dir: Path, record: Mapping[str, Any], name: str) -> Path:
    artifact = Path(name)
    step = record.get("step_idx")
    if (
        artifact.name != name
        or artifact.suffix.lower() != ".png"
        or type(step) is not int
        or step < 0
        or name not in record.get("artifacts", [])
    ):
        raise ValueError(f"invalid frame artifact reference: {name!r} at step {step!r}")
    return run_state_dir / name / f"{step:02d}.png"


def _last_with_artifact(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for record in reversed(records):
        if name in record.get("artifacts", []):
            return record
    raise ValueError(f"episode has no {name} frame")


def _frame_bytes(run_state_dir: Path, record: Mapping[str, Any], name: str) -> bytes:
    return _frame_path(run_state_dir, record, name).read_bytes()


def _task_language(records: list[dict[str, Any]], task_desc: Mapping[str, Any]) -> str:
    for record in records:
        extras = record.get("extras")
        if isinstance(extras, dict) and extras.get("task_language"):
            return str(extras["task_language"])
    return str(task_desc.get("task_language") or "")


def _suite_regime(task_desc: Mapping[str, Any]) -> tuple[str, str, str]:
    suite = str(task_desc["suite"])
    regime = str(task_desc.get("regime") or "")
    family = suite
    if not regime:
        prefix, separator, suffix = suite.rpartition("_")
        if separator and suffix in _REGIMES:
            family, regime = prefix, suffix
        else:
            regime = "task"
    return suite, regime, family.replace("_", "")


def _memory_ids(
    task_desc: Mapping[str, Any], *, family: str, regime: str, task_id: int
) -> list[str]:
    configured = task_desc.get("memory_ids")
    if configured is not None:
        return [str(value) for value in configured]
    return [f"task-family_{family}_{regime}_t{task_id}"]


class EpisodeWriter:
    def __init__(self, store: EpisodeStore) -> None:
        self._store = store

    def write(self, episode: Episode) -> str:
        payload = (episode.model_dump_json(indent=2) + "\n").encode("utf-8")
        return self._store.put_episode(
            memory_version=episode.memory_version,
            episode_id=episode.episode_id,
            payload=payload,
        )

    def write_from_run(
        self,
        *,
        run_state_dir: str | Path,
        cell_tag: str,
        robot: str,
        task_desc: Mapping[str, Any],
        outcome: str,
        stats: Mapping[str, Any],
        elapsed_s: float,
        memory_version: str | None = None,
        created_at: datetime | None = None,
    ) -> tuple[Episode, str]:
        run_state_dir = Path(run_state_dir).resolve()
        records = _records(run_state_dir)
        first = _last_with_artifact(list(reversed(records)), "agentview.png")
        first_uri = self._store.put_frame(
            _frame_bytes(run_state_dir, first, "agentview.png")
        )

        success_uri = None
        failure_uris: list[str] = []
        if outcome == "success":
            terminated = [
                record
                for record in records
                if record.get("terminated")
                and "agentview.png" in record.get("artifacts", [])
            ]
            if terminated:
                success_uri = self._store.put_frame(
                    _frame_bytes(run_state_dir, terminated[-1], "agentview.png")
                )
        elif outcome == "failure":
            truncated = [
                record
                for record in records
                if record.get("truncated")
                and "agentview.png" in record.get("artifacts", [])
            ]
            last = (
                truncated[-1]
                if truncated
                else _last_with_artifact(records, "agentview.png")
            )
            failure_uris = [
                self._store.put_frame(
                    _frame_bytes(run_state_dir, last, "agentview.png")
                )
            ]

        overlay_uris: list[str] = []
        for record in reversed(records):
            overlays = sorted(
                (
                    name
                    for name in record.get("artifacts", [])
                    if isinstance(name, str) and _OVERLAY.fullmatch(name)
                ),
                key=lambda name: int(_OVERLAY.fullmatch(name).group(1)),
                reverse=True,
            )
            if overlays:
                overlay_uris = [
                    self._store.put_frame(
                        _frame_bytes(run_state_dir, record, overlays[0])
                    )
                ]
                break

        created_at = created_at or datetime.fromtimestamp(
            (run_state_dir / "states.json").stat().st_mtime, UTC
        )
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        created_at = created_at.astimezone(UTC)
        suite, regime, family = _suite_regime(task_desc)
        task_id = int(task_desc.get("task_id", task_desc.get("task")))
        episode = Episode(
            episode_id=f"{cell_tag}_{created_at.strftime('%Y%m%dT%H%M%S')}",
            cell_tag=cell_tag,
            robot=robot,
            suite=suite,
            regime=regime,
            task_id=task_id,
            seed=task_desc.get("seed", 0),
            task_language=_task_language(records, task_desc),
            outcome=outcome,
            confidence="verified" if outcome == "success" else "single-shot",
            memory_ids=_memory_ids(
                task_desc, family=family, regime=regime, task_id=task_id
            ),
            memory_version=memory_version or DEFAULT_MEMORY_VERSION,
            embedding_model=EMBEDDING_MODEL_PLACEHOLDER,
            created_at=created_at,
            stats=EpisodeStats(
                elapsed_s=elapsed_s,
                tool_calls=stats.get("tool_calls", 0) or 0,
                output_tokens=stats.get(
                    "total_output_tokens", stats.get("output_tokens", 0)
                )
                or 0,
            ),
            assets=EpisodeAssets(
                first_frame=first_uri,
                success_frame=success_uri,
                failure_frames=failure_uris,
                seg_overlays=overlay_uris,
            ),
        )
        return episode, self.write(episode)
