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

import threading
import urllib.parse
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.libero import robot_spec
from rpent.robots.components.embedding_client import EmbeddingClient
from rpent.robots.components.embedding_server import (
    MOCK_DIM,
    EmbeddingFacade,
    read_image_uri,
)
from rpent.utils.config import EMBEDDING_MODEL_ID
from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc import RpcError, wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcClient


def test_mock_is_deterministic_and_matches_contract_dimension(tmp_path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"frame-content")
    facade = EmbeddingFacade(mock=True)

    first = facade.encode(text="拿起蓝色盘子", image_uris=[image.as_uri()])
    second = facade.encode(text="拿起蓝色盘子", image_uris=[image.as_uri()])
    different = facade.encode(text="拿起红色盘子", image_uris=[image.as_uri()])

    assert first == second
    assert first["model"] == EMBEDDING_MODEL_ID
    assert first["dim"] == MOCK_DIM == 1536
    assert len(first["vector"]) == MOCK_DIM
    assert np.linalg.norm(first["vector"]) == pytest.approx(1.0, abs=1e-6)
    assert first["vector"] != different["vector"]


def test_file_uri_resolution_uses_content_not_uri(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"same-image")
    second.write_bytes(b"same-image")
    facade = EmbeddingFacade(mock=True)

    assert read_image_uri(first.as_uri()) == b"same-image"
    assert facade.encode(image_uris=[first.as_uri()]) == facade.encode(
        image_uris=[second.as_uri()]
    )


def test_agfs_uri_resolution_reads_file_api(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b"agfs-image"

    def open(request, *, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open)
    uri = "agfs://frames/sha256/ab/cd/image.png"

    assert read_image_uri(uri, agfs_endpoint="http://agfs.local:8080") == b"agfs-image"
    parsed = urllib.parse.urlsplit(captured["url"])
    assert parsed.path == "/api/v1/files"
    assert urllib.parse.parse_qs(parsed.query) == {
        "path": ["/frames/sha256/ab/cd/image.png"]
    }
    assert captured["timeout"] > 0


def test_rejects_unsupported_uri_scheme() -> None:
    with pytest.raises(ValueError, match="unsupported image URI scheme"):
        read_image_uri("https://example.com/image.png")


@contextmanager
def _running_mock_server():
    facade = EmbeddingFacade(mock=True)
    port = pick_free_port()
    thread = threading.Thread(
        target=facade.serve,
        kwargs={"transport": "http", "host": "127.0.0.1", "port": port},
        daemon=True,
    )
    thread.start()
    rpc = HttpRpcClient(f"http://127.0.0.1:{port}")
    try:
        wait_for_ready(rpc, timeout_s=3.0, poll_interval_s=0.01)
        yield EmbeddingClient(f"127.0.0.1:{port}", retries=0)
    finally:
        if thread.is_alive():
            rpc.call("shutdown", timeout_s=1.0)
        thread.join(timeout=3.0)
        rpc.close()
    assert not thread.is_alive()


def test_post_encode_and_health_check_over_http() -> None:
    with _running_mock_server() as client:
        result = client.encode(text="blue plate")

    assert result["dim"] == MOCK_DIM
    assert result["model"] == EMBEDDING_MODEL_ID
    assert len(result["vector"]) == MOCK_DIM


class _FlakyRpc:
    def __init__(self, failures: list[Exception], response: dict[str, Any]) -> None:
        self._failures = failures
        self._response = response
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def call(self, method: str, *, kwargs: dict[str, Any], timeout_s: float):
        self.calls.append((method, kwargs, timeout_s))
        if self._failures:
            raise self._failures.pop(0)
        return self._response


def _response() -> dict[str, Any]:
    return {
        "vector": [0.0] * MOCK_DIM,
        "model": EMBEDDING_MODEL_ID,
        "dim": MOCK_DIM,
    }


def test_client_retries_transient_rpc_and_forwards_timeout() -> None:
    rpc = _FlakyRpc(
        [RpcError("embedding.encode", "unavailable"), TimeoutError("slow")],
        _response(),
    )
    client = EmbeddingClient(rpc, timeout_s=1.25, retries=2, retry_backoff_s=0)

    assert client.encode(text="plate") == _response()
    assert len(rpc.calls) == 3
    assert all(call[0] == "embedding.encode" for call in rpc.calls)
    assert all(call[2] == 1.25 for call in rpc.calls)


def test_client_stops_after_configured_retries() -> None:
    rpc = _FlakyRpc([TimeoutError("slow"), TimeoutError("slow")], _response())
    client = EmbeddingClient(rpc, timeout_s=0.5, retries=1, retry_backoff_s=0)

    with pytest.raises(TimeoutError, match="after 2 attempts"):
        client.encode(text="plate")
    assert len(rpc.calls) == 2


@pytest.mark.parametrize("record_episodes", [False, True])
def test_libero_runtime_starts_embedding_only_when_requested(
    monkeypatch, tmp_path, record_episodes
) -> None:
    started: list[str] = []
    args = SimpleNamespace(
        record_episodes=record_episodes,
        planner="api",
        collect_flywheel_data=False,
    )
    monkeypatch.setattr(
        robot_spec,
        "select_am",
        lambda *args: SimpleNamespace(is_wam=False),
    )

    def spawn(owned, events, component, starter):
        started.append(component)
        return None, component

    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *args, **kwargs: {})

    robot_spec._init_runtime(args, tmp_path, None, None)

    assert ("embedding" in started) is record_episodes


def test_libero_runtime_reuses_external_embedding_endpoint() -> None:
    args = SimpleNamespace(embedding_endpoint="http://127.0.0.1:8123")

    daemon, rpc = robot_spec._spawn_embedding_server(args, SimpleNamespace())

    assert daemon is None
    assert isinstance(rpc, HttpRpcClient)
    rpc.close()
