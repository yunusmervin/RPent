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

"""Best-effort write hooks for operator interactions."""

from __future__ import annotations

import threading
from typing import Any

from rpent.session_memory.service import DEFAULT_OPERATOR_ID
from rpent.utils.logging import get_logger

logger = get_logger("session_memory")


class SessionMemoryHook:
    """Record the first task and subsequent interactions only when writes are allowed."""

    def __init__(
        self,
        client: Any = None,
        *,
        user_id: str = DEFAULT_OPERATOR_ID,
        run_id: str,
        writable: bool,
    ) -> None:
        self.client = client
        self.user_id = user_id
        self.run_id = run_id
        self.writable = bool(writable and client is not None)
        self._task_recorded = False
        self._lock = threading.Lock()

    @classmethod
    def from_args(
        cls, args: Any, client: Any = None, *, run_id: str
    ) -> "SessionMemoryHook":
        """Apply the exploration/interaction gate and the HF read-only override."""
        interactive = bool(
            getattr(args, "tui", False) or getattr(args, "dashboard", False)
        )
        writable = bool(getattr(args, "explore", False) or interactive)
        writable &= getattr(args, "memory_profile", "hf") != "hf"
        return cls(
            client,
            user_id=getattr(args, "operator_id", DEFAULT_OPERATOR_ID),
            run_id=run_id,
            writable=writable,
        )

    def record_task(self, content: str) -> None:
        """Record the initial task once."""
        with self._lock:
            if self._task_recorded:
                return
            self._task_recorded = True
        self._write(content, "task")

    def record_user_message(self, content: str) -> None:
        """Treat the first submitted line as the task and later lines as steering."""
        with self._lock:
            event = "user_input" if self._task_recorded else "task"
            self._task_recorded = True
        self._write(content, event)

    def record_verdict(self, verdict: str) -> None:
        """Record accepted human success/failure verdicts, excluding abort controls."""
        if verdict in {"success", "failure"}:
            self._write(verdict, "verdict")

    def _write(self, content: str, event: str) -> None:
        if not self.writable or not isinstance(content, str) or not content.strip():
            return
        try:
            self.client.add(
                content=content,
                event=event,
                user_id=self.user_id,
                run_id=self.run_id,
            )
        except Exception as exc:
            logger.warning("session memory write skipped: %s", exc)
