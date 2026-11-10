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

"""Cross-session, mutable memory corpus."""

from typing import Any

from rpent.memory.agfs_backend import AGFSBackend
from rpent.memory.backend import MemoryBackend, get_memory_backend
from rpent.memory.manager import FileSystemBackend


def __getattr__(name: str) -> Any:
    if name == "MemoryManager":
        from rpent.memory.manager import MemoryManager

        return MemoryManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AGFSBackend",
    "FileSystemBackend",
    "MemoryBackend",
    "MemoryManager",
    "get_memory_backend",
]
