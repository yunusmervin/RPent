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

"""AGFS-mounted memory with a local content-addressed frame store."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from rpent.memory.manager import FileSystemBackend
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.logging import get_logger

logger = get_logger("memory.agfs")


class AGFSStartupError(RuntimeError):
    """The owned AGFS server could not be started or become ready."""


def _normalize_endpoint(endpoint: str) -> str:
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/")


def _wait_for_agfs_ready(
    endpoint: str,
    daemon: ProcessDaemon,
    *,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = daemon.poll()
        if returncode is not None:
            raise AGFSStartupError(
                f"agfs-server exited with code {returncode} before becoming ready"
            )
        try:
            with opener.open(f"{endpoint}/api/v1/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise AGFSStartupError(
        f"agfs-server did not become ready within {timeout_s:g}s: {last_error}"
    )


class AGFSBackend(FileSystemBackend):
    """Expose one robot's corpus, frames, episodes, and transcripts through AGFS.

    Corpus operations use the same local paths and authorization logic as
    :class:`FileSystemBackend`. The AGFS mounts therefore provide another view
    of the canonical files rather than a second source of truth.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        memory_access: str = "read_only",
        inbox_cell_tag: str | None = None,
        agfs_endpoint: str | None = None,
        frames_dir: str | Path | None = None,
        episodes_dir: str | Path | None = None,
        transcripts_dir: str | Path | None = None,
        robot: str | None = None,
        record_episodes: bool = False,
        server_bin: str = "agfs-server",
        startup_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(
            root,
            memory_access=memory_access,
            inbox_cell_tag=inbox_cell_tag,
        )
        self._robot = robot or self.root.name
        if not self._robot or self._robot in {".", ".."} or "/" in self._robot:
            raise ValueError(f"invalid AGFS robot name: {self._robot!r}")
        storage_root = self.root / "_internal" / "agfs"
        self._frames_dir = self._storage_path(frames_dir, storage_root / "frames")
        self._episodes_dir = self._storage_path(
            episodes_dir, storage_root / "episodes"
        )
        self._transcripts_dir = self._storage_path(
            transcripts_dir, storage_root / "transcripts"
        )
        self._episode_recording = record_episodes
        self._endpoint = (
            _normalize_endpoint(agfs_endpoint) if agfs_endpoint is not None else None
        )
        self._daemon: ProcessDaemon | None = None
        self._server_state: tempfile.TemporaryDirectory[str] | None = None
        if self._endpoint is None:
            port = pick_free_port()
            self._endpoint = f"http://127.0.0.1:{port}"
            self._start_server(
                port=port,
                server_bin=server_bin,
                startup_timeout_s=startup_timeout_s,
            )

    def _storage_path(self, configured: str | Path | None, default: Path) -> Path:
        path = Path(configured).resolve() if configured is not None else default.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"AGFS storage escapes robot memory root: {path}")
        return path

    @property
    def endpoint(self) -> str:
        """HTTP endpoint of the attached or owned AGFS server."""
        assert self._endpoint is not None
        return self._endpoint

    @property
    def daemon(self) -> ProcessDaemon | None:
        """The owned server process, or None when attached to an endpoint."""
        return self._daemon

    @property
    def frames_dir(self) -> Path:
        return self._frames_dir

    @property
    def episodes_dir(self) -> Path:
        return self._episodes_dir

    @property
    def transcripts_dir(self) -> Path:
        return self._transcripts_dir

    def _mounts(self) -> list[dict[str, Any]]:
        mounts = [
            (f"corpus-{self._robot}", f"/corpus/{self._robot}", self.root),
            ("frames", "/frames", self.frames_dir),
            ("transcripts", "/transcripts", self.transcripts_dir),
        ]
        if self._memory_access == "inbox_write" or self._episode_recording:
            mounts.append(("episodes", "/episodes", self.episodes_dir))
        return [
            {
                "name": name,
                "enabled": True,
                "path": mount_path,
                "config": {"local_dir": str(local_dir)},
            }
            for name, mount_path, local_dir in mounts
        ]

    def _start_server(
        self,
        *,
        port: int,
        server_bin: str,
        startup_timeout_s: float,
    ) -> None:
        state = tempfile.TemporaryDirectory(prefix="rpent-agfs-")
        self._server_state = state
        state_dir = Path(state.name)
        config_path = state_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "server": {"address": f"127.0.0.1:{port}", "log_level": "info"},
                    "plugins": {"localfs": self._mounts()},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "rpent.memory.agfs_server",
            "--server-bin",
            server_bin,
            "--config",
            str(config_path),
            "--address",
            f"127.0.0.1:{port}",
            "--parent-watch",
        ]
        daemon = ProcessDaemon(
            "agfs-server",
            command,
            log_path=str(state_dir / "server.log"),
        )
        self._daemon = daemon
        try:
            if self._memory_access == "inbox_write" or self._episode_recording:
                self.frames_dir.mkdir(parents=True, exist_ok=True)
                self.episodes_dir.mkdir(parents=True, exist_ok=True)
            if self._memory_access == "inbox_write":
                self.transcripts_dir.mkdir(parents=True, exist_ok=True)
            daemon.start()
            _wait_for_agfs_ready(
                self.endpoint,
                daemon,
                timeout_s=startup_timeout_s,
            )
        except Exception as exc:
            daemon.stop()
            state.cleanup()
            self._server_state = None
            self._daemon = None
            if isinstance(exc, AGFSStartupError):
                raise
            raise AGFSStartupError(f"failed to start agfs-server: {exc}") from exc
        logger.info("AGFS ready at %s for robot %s", self.endpoint, self._robot)

    def enable_episode_recording(self) -> None:
        """Allow frame and episode writes without enabling corpus writes."""
        self._episode_recording = True

    def put_frame(self, png_bytes: bytes) -> str:
        """Store PNG bytes by SHA256 and return their stable AGFS URI."""
        if self._memory_access != "inbox_write" and not self._episode_recording:
            raise PermissionError("writing frames is denied in this mode")
        if not isinstance(png_bytes, bytes):
            raise TypeError("png_bytes must be bytes")
        digest = hashlib.sha256(png_bytes).hexdigest()
        relative = Path("sha256") / digest[:2] / digest[2:4] / f"{digest}.png"
        destination = self.frames_dir / relative
        uri = f"agfs://frames/{relative.as_posix()}"
        if destination.exists():
            return uri
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{digest}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(png_bytes)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return uri

    def put_episode(
        self,
        *,
        memory_version: str,
        episode_id: str,
        payload: bytes,
    ) -> str:
        """Atomically store one canonical episode JSON document."""
        if self._memory_access != "inbox_write" and not self._episode_recording:
            raise PermissionError("writing episodes is denied in this mode")
        for component in (memory_version, episode_id):
            if (
                not component
                or component in {".", ".."}
                or Path(component).name != component
                or "/" in component
                or "\\" in component
            ):
                raise ValueError(f"invalid episode path component: {component!r}")
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        destination = self.episodes_dir / memory_version / f"{episode_id}.json"
        uri = f"agfs://episodes/{memory_version}/{episode_id}.json"
        if destination.exists():
            return uri
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{episode_id}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return uri

    def close(self) -> None:
        """Stop only the AGFS process owned by this backend."""
        if self._daemon is not None:
            self._daemon.stop()
            self._daemon = None
        if self._server_state is not None:
            self._server_state.cleanup()
            self._server_state = None


__all__ = ["AGFSBackend", "AGFSStartupError"]
