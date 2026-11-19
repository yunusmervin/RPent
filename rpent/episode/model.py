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

"""Persistent episode metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EpisodeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EpisodeStats(EpisodeModel):
    elapsed_s: float = Field(ge=0)
    tool_calls: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


def _validate_frame_uri(value: str) -> str:
    if not value.startswith("agfs://frames/sha256/"):
        raise ValueError("frame assets must use content-addressed AGFS URIs")
    if "base64," in value:
        raise ValueError("frame assets must not inline image bytes")
    return value


class EpisodeAssets(EpisodeModel):
    first_frame: str
    success_frame: str | None = None
    failure_frames: list[str] = Field(default_factory=list)
    seg_overlays: list[str] = Field(default_factory=list)

    @field_validator("first_frame", "success_frame")
    @classmethod
    def validate_optional_frame(cls, value: str | None) -> str | None:
        return _validate_frame_uri(value) if value is not None else None

    @field_validator("failure_frames", "seg_overlays")
    @classmethod
    def validate_frame_list(cls, values: list[str]) -> list[str]:
        return [_validate_frame_uri(value) for value in values]


class Episode(EpisodeModel):
    episode_id: str = Field(min_length=1, pattern=r"^[^/\\]+$")
    cell_tag: str = Field(min_length=1)
    robot: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    task_id: int = Field(ge=0)
    seed: int = Field(ge=0)
    task_language: str = Field(min_length=1)
    outcome: Literal["success", "failure", "aborted"]
    confidence: Literal["single-shot", "probable", "verified"]
    memory_ids: list[str] = Field(default_factory=list)
    memory_version: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    created_at: datetime
    stats: EpisodeStats
    assets: EpisodeAssets

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @field_validator("memory_ids")
    @classmethod
    def validate_memory_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("memory_ids must not contain empty identifiers")
        return values
