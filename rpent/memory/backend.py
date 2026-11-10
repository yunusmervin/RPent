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

"""Memory backend contract and implementation factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from rpent.memory.agfs_backend import AGFSBackend, AGFSStartupError
from rpent.memory.manager import FileSystemBackend
from rpent.utils.logging import get_logger

logger = get_logger("memory.backend")


class MemoryBackend(Protocol):
    """Memory backend protocol implemented by the filesystem backend."""

    @property
    def root(self) -> Path: ...

    def authorize_read(self, path: str | Path) -> Path: ...

    def authorize_write(self, path: str | Path) -> Path: ...

    def list_directory(self, path: str | Path) -> tuple[Path, list[str] | None]: ...

    def record_read(self, path: Path, content: str) -> None: ...

    def merge_memory(
        self,
        *,
        cell_tag: str,
        run_state_dir: str | Path,
        solved: bool,
    ) -> dict[str, Any]: ...

    def rebuild_index(self) -> Path | None: ...

    def validate(self) -> list[str]: ...

    def sync(
        self,
        *,
        remote_repo: str,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path: ...


def get_memory_backend(name: str = "filesystem", **kwargs) -> MemoryBackend:
    backend_class = kwargs.pop("_backend_class", FileSystemBackend)
    agfs_endpoint = kwargs.pop("agfs_endpoint", None)
    robot = kwargs.pop("robot", None)
    if name == "filesystem":
        return backend_class(**kwargs)
    if name == "agfs":
        if agfs_endpoint is not None:
            kwargs["agfs_endpoint"] = agfs_endpoint
        if robot is not None:
            kwargs["robot"] = robot
        try:
            return AGFSBackend(**kwargs)
        except AGFSStartupError as exc:
            logger.warning(
                "AGFS startup failed; falling back to FileSystemBackend: %s", exc
            )
            fallback_keys = {"root", "memory_access", "inbox_cell_tag"}
            fallback_kwargs = {
                key: value for key, value in kwargs.items() if key in fallback_keys
            }
            return backend_class(**fallback_kwargs)
    raise ValueError(f"unknown memory backend: {name!r}")


__all__ = ["MemoryBackend", "get_memory_backend"]
