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

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest
import yaml

from rpent.cli.main import _build_argparser
from rpent.memory import AGFSBackend, FileSystemBackend, get_memory_backend
from rpent.memory import backend as backend_module
from rpent.memory.agfs_backend import AGFSStartupError


def _backend(name: str, root: Path, **kwargs):
    if name == "agfs":
        kwargs.update(agfs_endpoint="http://127.0.0.1:18080", robot="libero")
    return get_memory_backend(name, root=root, **kwargs)


@pytest.mark.parametrize("backend_name", ["filesystem", "agfs"])
def test_filesystem_and_agfs_share_authorization_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
) -> None:
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    root = tmp_path / "memory" / "libero"
    published = root / "global" / "strategy.md"
    published.parent.mkdir(parents=True)
    published.write_text("strategy", encoding="utf-8")
    foreign = tmp_path / "memory" / "robotwin" / "global" / "foreign.md"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("foreign", encoding="utf-8")
    backend = _backend(
        backend_name,
        root,
        memory_access="inbox_write",
        inbox_cell_tag="current-cell",
    )

    assert backend.root == root.resolve()
    assert backend.authorize_read(published) == published.resolve()
    assert backend.record_read(published, "strategy") is None
    assert backend.list_directory(root / "global") == (
        root / "global",
        ["strategy.md"],
    )
    own_inbox = root / "_internal" / "inbox" / "current-cell" / "draft.md"
    other_inbox = root / "_internal" / "inbox" / "other-cell" / "draft.md"
    assert backend.authorize_read(own_inbox) == own_inbox.resolve()
    assert backend.authorize_write(own_inbox) == own_inbox.resolve()
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        backend.authorize_write(root / "global" / "new.md")
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        backend.authorize_write(other_inbox)
    with pytest.raises(PermissionError, match="reading this memory path is denied"):
        backend.authorize_read(other_inbox)
    with pytest.raises(PermissionError, match="another robot's memory is denied"):
        backend.authorize_read(foreign)


@pytest.mark.parametrize("backend_name", ["filesystem", "agfs"])
def test_filesystem_and_agfs_share_merge_and_index_contract(
    tmp_path: Path,
    backend_name: str,
) -> None:
    root = tmp_path / "memory" / "libero"
    cell = "libero10_task_t2_s0"
    draft = root / "_internal" / "inbox" / cell / "global_strategy.md"
    draft.parent.mkdir(parents=True)
    draft.write_text(
        "---\n"
        "id: global_strategy\n"
        "scope: global\n"
        "kind: strategy\n"
        "title: Reliable strategy\n"
        "applies_when: the scene matches\n"
        "evidence:\n"
        f"  cells: [{cell}]\n"
        "  attempts: 1\n"
        "  solved_seeds: [0]\n"
        "  failed_seeds: []\n"
        "confidence: single-shot\n"
        "related: []\n"
        "---\n"
        "Contract body.\n",
        encoding="utf-8",
    )
    backend = _backend(backend_name, root)

    result = backend.merge_memory(
        cell_tag=cell,
        run_state_dir=tmp_path / "run",
        solved=False,
    )

    assert result["global"] == 1
    assert (root / "global" / "global_strategy.md").is_file()
    assert backend.rebuild_index() == root / "MEMORY.md"
    assert "global_strategy.md" in (root / "MEMORY.md").read_text(encoding="utf-8")
    assert backend.validate() == []


def test_agfs_put_frame_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "memory" / "libero"
    backend = _backend(
        "agfs",
        root,
        memory_access="inbox_write",
        inbox_cell_tag="cell",
    )
    png = b"\x89PNG\r\n\x1a\nframe-content"

    first = backend.put_frame(png)
    second = backend.put_frame(png)

    digest = hashlib.sha256(png).hexdigest()
    expected = f"agfs://frames/sha256/{digest[:2]}/{digest[2:4]}/{digest}.png"
    assert first == second == expected
    frame = root / "_internal" / "agfs" / "frames" / "sha256"
    frame = frame / digest[:2] / digest[2:4] / f"{digest}.png"
    assert frame.read_bytes() == png
    assert len(list(frame.parent.glob("*.tmp"))) == 0
    backend.close()


def test_agfs_frame_writes_are_read_only_during_evaluation(tmp_path: Path) -> None:
    backend = _backend("agfs", tmp_path / "memory" / "libero")

    with pytest.raises(PermissionError, match="writing frames is denied"):
        backend.put_frame(b"png")

    assert not (backend.root / "_internal").exists()


def test_agfs_external_endpoint_does_not_start_a_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_start(*args, **kwargs) -> None:
        raise AssertionError("an external endpoint must not start agfs-server")

    monkeypatch.setattr("rpent.memory.agfs_backend.ProcessDaemon.start", fail_start)
    backend = AGFSBackend(
        tmp_path / "memory" / "libero",
        agfs_endpoint="127.0.0.1:18080",
    )

    assert backend.endpoint == "http://127.0.0.1:18080"
    assert backend.daemon is None


def test_agfs_local_server_mounts_and_parent_watch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []

    class FakeDaemon:
        def __init__(self, name, cmd, *, log_path=None) -> None:
            self.name = name
            self.cmd = cmd
            self.log_path = log_path
            self.started = False
            self.stopped = False
            created.append(self)

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def poll(self):
            return None

    monkeypatch.setattr("rpent.memory.agfs_backend.ProcessDaemon", FakeDaemon)
    monkeypatch.setattr("rpent.memory.agfs_backend.pick_free_port", lambda: 23456)
    monkeypatch.setattr(
        "rpent.memory.agfs_backend._wait_for_agfs_ready", lambda *args, **kwargs: None
    )
    backend = AGFSBackend(tmp_path / "memory" / "libero", robot="libero")

    daemon = created[0]
    assert daemon.started
    assert "--parent-watch" in daemon.cmd
    config_path = Path(daemon.cmd[daemon.cmd.index("--config") + 1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["server"]["address"] == "127.0.0.1:23456"
    mounts = {
        item["path"]: Path(item["config"]["local_dir"])
        for item in config["plugins"]["localfs"]
    }
    assert mounts == {
        "/corpus/libero": backend.root,
        "/frames": backend.frames_dir,
        "/transcripts": backend.transcripts_dir,
    }
    backend.close()
    assert daemon.stopped


def test_agfs_startup_failure_falls_back_to_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenAGFS:
        def __init__(self, **kwargs) -> None:
            raise AGFSStartupError("server unavailable")

    monkeypatch.setattr(backend_module, "AGFSBackend", BrokenAGFS)
    with caplog.at_level(logging.WARNING, logger="rpent.memory.backend"):
        backend = get_memory_backend(
            "agfs",
            root=tmp_path / "memory" / "libero",
            memory_access="inbox_write",
            inbox_cell_tag="cell",
        )

    assert isinstance(backend, FileSystemBackend)
    assert "falling back to FileSystemBackend" in caplog.text
    assert "server unavailable" in caplog.text


def test_agfs_cli_options_default_to_filesystem() -> None:
    parser = _build_argparser()
    defaults = parser.parse_args([])
    configured = parser.parse_args(
        ["--memory-backend", "agfs", "--agfs-endpoint", "127.0.0.1:18080"]
    )

    assert defaults.memory_backend == "filesystem"
    assert defaults.agfs_endpoint is None
    assert configured.memory_backend == "agfs"
    assert configured.agfs_endpoint == "127.0.0.1:18080"
