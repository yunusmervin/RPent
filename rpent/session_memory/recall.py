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

"""Preference recall, prompt formatting, and the readonly planner tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rpent.tools.base import ToolResult, iter_tools, tool
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.tools.toolkit import Toolkit

logger = get_logger("session_memory")


def _memory_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("memory", "text", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _format_records(records: list[Any]) -> str:
    return "\n".join(f"- {text}" for item in records if (text := _memory_text(item)))


def recall_preference_records(
    client: Any,
    query: str,
    *,
    user_id: str,
    run_id: str,
    top_k: int = 5,
) -> list[Any]:
    """Recall preferences without allowing service failures to stop a run."""
    if client is None or not isinstance(query, str) or not query.strip():
        return []
    try:
        response = client.search(
            query=query,
            user_id=user_id,
            run_id=run_id,
            top_k=top_k,
        )
        records = response.get("results", [])
        if not isinstance(records, list):
            raise RuntimeError("session memory search returned an invalid result")
        return records
    except Exception as exc:
        logger.warning("session memory recall unavailable: %s", exc)
        return []


def recall_preferences(
    client: Any,
    query: str,
    *,
    user_id: str,
    run_id: str,
    top_k: int = 5,
) -> str:
    """Return the top preferences as a prompt-ready Markdown list."""
    records = recall_preference_records(
        client,
        query,
        user_id=user_id,
        run_id=run_id,
        top_k=top_k,
    )
    return _format_records(records)


class PreferenceRecallTools:
    """Bind the readonly preference tool to one operator and cell."""

    def __init__(self, client: Any, *, user_id: str, run_id: str) -> None:
        self.client = client
        self.user_id = user_id
        self.run_id = run_id

    @tool(readonly=True)
    def recall_preferences(self, topic: str, top_k: int = 5) -> ToolResult:
        """Recall the current operator's preferences and confirmation habits."""
        records = recall_preference_records(
            self.client,
            topic,
            user_id=self.user_id,
            run_id=self.run_id,
            top_k=top_k,
        )
        return ToolResult(
            data={
                "operator_preferences": _format_records(records),
                "results": records,
            }
        )


def register_preference_recall(
    toolkit: "Toolkit", client: Any, *, user_id: str, run_id: str
) -> None:
    """Expose preference recall only when the optional service is configured."""
    if client is None:
        return
    owner = PreferenceRecallTools(client, user_id=user_id, run_id=run_id)
    toolkit.add_tools(iter_tools(owner))
