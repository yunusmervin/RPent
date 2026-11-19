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

"""Persistent freeze guards shared by memory backends and maintenance tools."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path("_internal/freeze/agfs-readonly.json")


def _component(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"invalid frozen version: {value!r}")


def _within_root(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"freeze path must be absolute: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"freeze path escapes robot memory root: {candidate}")
    return resolved


def _validate_timestamp(value: Any, version: str) -> None:
    try:
        frozen_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if frozen_at.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be UTC")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid freeze timestamp for {version!r}") from exc


def read_freeze_config(root: str | Path) -> dict[str, Any]:
    """Read and validate one robot's backend freeze configuration.

    A missing configuration means the corpus has not been frozen. Malformed
    or escaping configuration raises rather than silently allowing a write.
    """
    root = Path(root).resolve()
    configured = root / _CONFIG_PATH
    path = _within_root(root, configured)
    if not path.exists():
        if configured.is_symlink():
            raise ValueError(f"invalid freeze configuration symlink: {configured}")
        return {"schema_version": 1, "versions": {}}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid freeze configuration: {path}") from exc
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or not isinstance(config.get("versions"), dict)
    ):
        raise ValueError(f"invalid freeze configuration schema: {path}")
    for version, entry in config["versions"].items():
        _component(version)
        if not isinstance(entry, dict):
            raise ValueError(f"invalid freeze entry for {version!r}")
        for field in ("corpus_root", "episodes_dir"):
            if not isinstance(entry.get(field), str):
                raise ValueError(f"freeze entry requires {field!r}: {version!r}")
            _within_root(root, entry[field])
        if Path(entry["episodes_dir"]).name != version:
            raise ValueError(
                f"episode directory does not match frozen version {version!r}"
            )
        _validate_timestamp(entry.get("frozen_at"), version)
    return config


def freeze_mounts(
    root: str | Path,
    version: str,
    corpus_root: str | Path,
    episodes_dir: str | Path,
    frozen_at: str,
) -> None:
    """Persist readonly corpus and episode mounts without editing corpus prose."""
    root = Path(root).resolve()
    _component(version)
    _validate_timestamp(frozen_at, version)
    corpus = _within_root(root, corpus_root)
    episodes = _within_root(root, episodes_dir)
    if episodes.name != version:
        raise ValueError("episodes_dir must be the version directory")
    config = read_freeze_config(root)
    entry = {
        "corpus_root": str(corpus),
        "episodes_dir": str(episodes),
        "frozen_at": frozen_at,
    }
    previous = config["versions"].get(version)
    if previous is not None and previous != entry:
        raise RuntimeError(
            f"version {version!r} already has a different freeze configuration"
        )
    config["versions"][version] = entry
    path = _within_root(root, root / _CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=path.parent, prefix=".agfs-readonly-", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    read_freeze_config(root)


def _freeze_configs(root: Path) -> Iterator[dict[str, Any]]:
    """Find robot metadata when a runtime uses a nested version corpus root."""
    for candidate in (root, *root.parents):
        if candidate != root and candidate.name == "memory":
            break
        config_path = candidate / _CONFIG_PATH
        if candidate == root or config_path.exists() or config_path.is_symlink():
            yield read_freeze_config(candidate)


def assert_corpus_writable(root: str | Path, *, recursive: bool = False) -> None:
    """Reject writes to a frozen corpus, including backends created before freeze."""
    root = Path(root).resolve()
    for config in _freeze_configs(root):
        for version, entry in config["versions"].items():
            corpus = Path(entry["corpus_root"]).resolve()
            if root.is_relative_to(corpus) or (
                recursive and corpus.is_relative_to(root)
            ):
                raise PermissionError(
                    f"memory corpus is frozen for version {version!r}"
                )


def assert_episode_writable(root: str | Path, version: str) -> None:
    """Reject writes to any episode version frozen in this robot's metadata."""
    _component(version)
    for config in _freeze_configs(Path(root).resolve()):
        if version in config["versions"]:
            raise PermissionError(f"episode version {version!r} is frozen")
