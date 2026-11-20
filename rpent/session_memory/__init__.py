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

"""Operator session history and preferences, isolated from task knowledge memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rpent.session_memory.hook import SessionMemoryHook
    from rpent.session_memory.recall import recall_preferences
    from rpent.session_memory.service import (
        DEFAULT_OPERATOR_ID,
        SESSION_MEMORY_COLLECTION,
        SessionMemoryClient,
        SessionMemoryService,
        create_session_memory_client,
    )

__all__ = [
    "DEFAULT_OPERATOR_ID",
    "SESSION_MEMORY_COLLECTION",
    "SessionMemoryClient",
    "SessionMemoryHook",
    "SessionMemoryService",
    "create_session_memory_client",
    "recall_preferences",
]


def __getattr__(name: str) -> Any:
    if name == "SessionMemoryHook":
        from rpent.session_memory.hook import SessionMemoryHook

        return SessionMemoryHook
    if name == "recall_preferences":
        from rpent.session_memory.recall import recall_preferences

        return recall_preferences
    if name in {
        "DEFAULT_OPERATOR_ID",
        "SESSION_MEMORY_COLLECTION",
        "SessionMemoryClient",
        "SessionMemoryService",
        "create_session_memory_client",
    }:
        from rpent.session_memory import service

        return getattr(service, name)
    raise AttributeError(name)
