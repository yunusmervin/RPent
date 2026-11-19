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

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rpent.episode.model import Episode


def _episode() -> dict:
    return {
        "episode_id": "libero10_swap_t2_s0_20261110T031500",
        "cell_tag": "libero10_swap_t2_s0",
        "robot": "libero",
        "suite": "libero_10_swap",
        "regime": "swap",
        "task_id": 2,
        "seed": 0,
        "task_language": "pick up the akita black bowl",
        "outcome": "success",
        "confidence": "verified",
        "memory_ids": ["task-family_libero10_swap_t2"],
        "memory_version": "unversioned",
        "embedding_model": "pending",
        "created_at": datetime(2026, 11, 10, 3, 15, tzinfo=UTC),
        "stats": {"elapsed_s": 41.2, "tool_calls": 17, "output_tokens": 0},
        "assets": {
            "first_frame": "agfs://frames/sha256/ab/cd/first.png",
            "success_frame": "agfs://frames/sha256/ab/cd/success.png",
            "failure_frames": [],
            "seg_overlays": [],
        },
    }


@pytest.mark.parametrize("field", ["episode_id", "outcome", "stats", "assets"])
def test_episode_requires_schema_fields(field: str) -> None:
    payload = _episode()
    del payload[field]

    with pytest.raises(ValidationError):
        Episode(**payload)


def test_episode_rejects_invalid_outcome() -> None:
    payload = _episode()
    payload["outcome"] = "solved"

    with pytest.raises(ValidationError, match="outcome"):
        Episode(**payload)


def test_episode_rejects_naive_timestamp_and_inline_images() -> None:
    payload = _episode()
    payload["created_at"] = datetime(2026, 11, 10, 3, 15)
    with pytest.raises(ValidationError, match="timezone"):
        Episode(**payload)

    payload = _episode()
    payload["assets"]["first_frame"] = "data:image/png;base64,AAAA"
    with pytest.raises(ValidationError, match="first_frame"):
        Episode(**payload)
