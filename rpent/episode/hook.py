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

"""Failure-isolated episode recording for run finalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.episode.writer import EpisodeStore, EpisodeWriter
from rpent.memory.freeze_state import assert_episode_writable

if TYPE_CHECKING:
    from rpent.episode.milvus_sink import EpisodeEncoder
from rpent.utils.logging import get_logger

logger = get_logger("episode")


def _outcome(
    finish_result: Mapping[str, Any] | None,
    environment_success: bool | None,
) -> str:
    finish_result = finish_result or {}
    status = str(finish_result.get("status") or "").lower()
    if finish_result.get("operator_aborted") or status == "aborted":
        return "aborted"
    if environment_success is True or status == "success":
        return "success"
    return "failure"


def _task_desc(
    run_state_dir: Path, cell_tag: str, task_desc: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(task_desc)
    audit_path = run_state_dir / f"{cell_tag}.json"
    if audit_path.is_file():
        with audit_path.open(encoding="utf-8") as stream:
            audit = json.load(stream)
        if isinstance(audit, dict):
            merged.update(audit)
    return merged


def record_episode(
    store: EpisodeStore,
    *,
    run_state_dir: str | Path,
    cell_tag: str,
    robot: str,
    task_desc: Mapping[str, Any],
    finish_result: Mapping[str, Any] | None,
    stats: Mapping[str, Any] | None,
    elapsed_s: float,
    memory_version: str | None = None,
    environment_success: bool | None = None,
    allow_read_only_write: bool = False,
    created_at: datetime | None = None,
    embedding_client: EpisodeEncoder | None = None,
    milvus_uri: str | None = None,
) -> str | None:
    try:
        root = getattr(store, "root", None)
        if root is not None and memory_version is not None:
            assert_episode_writable(root, memory_version)
        if allow_read_only_write:
            enable = getattr(store, "enable_episode_recording", None)
            if callable(enable):
                enable()
        run_state_dir = Path(run_state_dir)
        episode, uri = EpisodeWriter(store).write_from_run(
            run_state_dir=run_state_dir,
            cell_tag=cell_tag,
            robot=robot,
            task_desc=_task_desc(run_state_dir, cell_tag, task_desc),
            outcome=_outcome(finish_result, environment_success),
            stats=stats or {},
            elapsed_s=elapsed_s,
            memory_version=memory_version,
            created_at=created_at,
        )
        if embedding_client is not None:
            try:
                from rpent.episode.milvus_sink import MilvusEpisodeSink

                sink = MilvusEpisodeSink(
                    embedding_client,
                    memory_root=store.root,
                    milvus_uri=milvus_uri,
                )
                try:
                    sink.upsert(episode, uri)
                finally:
                    sink.close()
            except Exception as exc:
                logger.warning(
                    "episode indexing failed; canonical episode remains at %s: %s: %s",
                    uri,
                    type(exc).__name__,
                    exc,
                )
        logger.info("episode recorded: %s (%s)", episode.episode_id, uri)
        return uri
    except Exception as exc:
        logger.warning(
            "episode recording failed; run result is unchanged: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None
