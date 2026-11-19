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

"""Independent HTTP service for Qwen3-VL text and image embeddings."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rpent.utils.config import EMBEDDING_MODEL_ID
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade
from rpent.utils.rpc.rpc_facade import make_error_response

logger = get_logger("embedding_server")

MOCK_DIM = 1536
_IMAGE_TIMEOUT_S = 30.0


def _validate_input(
    text: str | None, image_uris: list[str] | None
) -> tuple[str, list[str]]:
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if image_uris is None:
        image_uris = []
    if not isinstance(image_uris, list) or not all(
        isinstance(uri, str) and uri for uri in image_uris
    ):
        raise ValueError("image_uris must be a list of non-empty strings")
    if not text and not image_uris:
        raise ValueError("encode requires text or at least one image URI")
    return text, image_uris


def _read_file_uri(uri: str) -> bytes:
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"invalid file image URI: {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError("file image URI must not contain a query or fragment")
    path = Path(urllib.request.url2pathname(parsed.path))
    return path.read_bytes()


def _read_agfs_uri(uri: str, endpoint: str | None) -> bytes:
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme != "agfs" or not parsed.netloc or not parsed.path:
        raise ValueError(f"invalid AGFS image URI: {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError("AGFS image URI must not contain a query or fragment")
    if not endpoint:
        raise ValueError(
            "reading agfs:// image URIs requires --agfs-endpoint or AGFS_ENDPOINT"
        )
    path = urllib.parse.unquote(parsed.path)
    query = urllib.parse.urlencode({"path": f"/{parsed.netloc}{path}"})
    url = f"{endpoint.rstrip('/')}/api/v1/files?{query}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_IMAGE_TIMEOUT_S) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"failed to read AGFS image {uri!r}: {exc}") from exc


def read_image_uri(uri: str, *, agfs_endpoint: str | None = None) -> bytes:
    """Read an image from a ``file://`` or ``agfs://`` URI."""
    if not isinstance(uri, str):
        raise ValueError("image URI must be a string")
    scheme = urllib.parse.urlsplit(uri).scheme
    if scheme == "file":
        return _read_file_uri(uri)
    if scheme == "agfs":
        return _read_agfs_uri(uri, agfs_endpoint)
    raise ValueError(f"unsupported image URI scheme {scheme!r}; expected file or agfs")


def _mock_vector(text: str, images: list[bytes]) -> list[float]:
    digest = hashlib.sha256()
    for kind, content in [
        ("text", text.encode("utf-8")),
        *(("image", image) for image in images),
    ]:
        encoded_kind = kind.encode("ascii")
        digest.update(len(encoded_kind).to_bytes(4, "big"))
        digest.update(encoded_kind)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    seed = int.from_bytes(digest.digest(), byteorder="big", signed=False)
    vector = np.random.Generator(np.random.PCG64(seed)).standard_normal(MOCK_DIM)
    vector /= np.linalg.norm(vector)
    return vector.astype(np.float32).tolist()


class _EmbeddingHttpHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            request = json.loads(body)
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            if self.path == "/encode":
                result = self.server.facade.encode(  # type: ignore[attr-defined]
                    text=request.get("text", ""),
                    image_uris=request.get("image_uris", []),
                )
                self._send_json(200, result)
                return
            if self.path == "/call":
                result = self.server.dispatch(  # type: ignore[attr-defined]
                    request["method"],
                    tuple(request.get("args", [])),
                    request.get("kwargs", {}),
                    session_id=request.get("session_id"),
                )
                self._send_json(200, {"ok": True, "result": result})
                return
            self._send_json(404, {"error": "not found"})
        except Exception as exc:
            if self.path == "/call":
                self._send_json(200, make_error_response(exc))
            else:
                status = 400 if isinstance(exc, ValueError) else 500
                self._send_json(status, {"error": str(exc)})


class _EmbeddingHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], facade: EmbeddingFacade):
        self.facade = facade
        self.dispatch = facade._dispatch
        super().__init__(server_address, _EmbeddingHttpHandler)


class EmbeddingFacade(RpcFacade):
    """Serve text and image embeddings from one model in a separate process."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        mock: bool = False,
        agfs_endpoint: str | None = None,
    ) -> None:
        super().__init__()
        self._mock = mock
        self._agfs_endpoint = agfs_endpoint
        self._model = None
        self._lock = threading.Lock()
        if mock:
            self._dim = MOCK_DIM
        else:
            if not model_path:
                raise ValueError(
                    "embedding model path is required unless --mock is enabled"
                )
            self._load(model_path)
        self._rpc["embedding.encode"] = self.encode
        self._readonly_methods.add("embedding.encode")
        logger.info(
            "embedding model ready: model=%s, dim=%d, mock=%s",
            EMBEDDING_MODEL_ID,
            self._dim,
            self._mock,
        )

    def _load(self, model_path: str) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "embedding dependencies are missing; install RPent with "
                '`pip install -e ".[embedding]"` or use --mock'
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading embedding model from %s on %s", model_path, device)
        self._model = SentenceTransformer(model_path, device=device)
        self._dim = MOCK_DIM

    def _encode_real(self, text: str, images: list[bytes]) -> list[float]:
        from PIL import Image

        model_input: dict[str, Any] = {}
        if text:
            model_input["text"] = text
        if images:
            decoded = [Image.open(io.BytesIO(image)).convert("RGB") for image in images]
            model_input["image"] = decoded if len(decoded) > 1 else decoded[0]
        with self._lock:
            vectors = self._model.encode(
                [model_input],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                truncate_dim=MOCK_DIM,
            )
        vector = np.asarray(vectors, dtype=np.float32)
        if vector.shape != (1, MOCK_DIM):
            raise RuntimeError(
                f"embedding model returned shape {vector.shape}; expected (1, {MOCK_DIM})"
            )
        return vector[0].tolist()

    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]:
        """Encode text and images into the shared 1536-dimensional space."""
        text, image_uris = _validate_input(text, image_uris)
        images = [
            read_image_uri(uri, agfs_endpoint=self._agfs_endpoint) for uri in image_uris
        ]
        if not images and not text:
            raise ValueError("encode requires non-empty text or image content")
        if any(not image for image in images):
            raise ValueError("image content must not be empty")
        vector = (
            _mock_vector(text, images)
            if self._mock
            else self._encode_real(text, images)
        )
        return {"vector": vector, "model": EMBEDDING_MODEL_ID, "dim": self._dim}

    def _bind_and_announce(
        self,
        transport: Literal["socket", "http"],
        host: str,
        port: int,
        dispatch: Any,
    ) -> _EmbeddingHttpServer:
        if transport != "http":
            raise ValueError("embedding_server supports HTTP transport only")
        del dispatch
        server = _EmbeddingHttpServer((host, port), self)
        bound_host, bound_port = server.server_address
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"http://{client_host}:{bound_port}"
        print(f"RPC server listening on {url}", flush=True)
        logger.info("RPC server listening on %s", url)
        return server


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPent Qwen3-VL embedding server")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--embedding-model-path",
        default=os.environ.get("EMBEDDING_MODEL_PATH"),
        help="Local model path or model repository (defaults to EMBEDDING_MODEL_PATH).",
    )
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument(
        "--agfs-endpoint",
        default=os.environ.get("AGFS_ENDPOINT"),
        help="AGFS HTTP endpoint used to read agfs:// image URIs.",
    )
    parser.add_argument("--parent-watch", action="store_true")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    facade = EmbeddingFacade(
        model_path=args.embedding_model_path,
        mock=args.mock,
        agfs_endpoint=args.agfs_endpoint,
    )
    facade.serve(
        transport="http",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
