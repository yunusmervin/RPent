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

"""Retrying client for the independent multimodal embedding service."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from rpent.utils.config import EMBEDDING_MODEL_ID
from rpent.utils.rpc import RpcClient, RpcError


class EmbeddingClient:
    """Call ``POST /encode`` directly or through an existing RPC client."""

    def __init__(
        self,
        endpoint: str | RpcClient,
        *,
        timeout_s: float = 120.0,
        retries: int = 2,
        retry_backoff_s: float = 0.1,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if retries < 0:
            raise ValueError("retries must be nonnegative")
        if retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be nonnegative")
        self._rpc = (
            endpoint
            if isinstance(endpoint, RpcClient)
            or callable(getattr(endpoint, "call", None))
            else None
        )
        self._endpoint = None if self._rpc is not None else self._normalize(endpoint)
        self._timeout_s = float(timeout_s)
        self._retries = int(retries)
        self._retry_backoff_s = float(retry_backoff_s)
        self._opener = None
        if self._endpoint is not None:
            hostname = urllib.parse.urlsplit(self._endpoint).hostname
            if hostname in {"127.0.0.1", "localhost"}:
                self._opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({})
                )

    @staticmethod
    def _normalize(endpoint: Any) -> str:
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError(
                "embedding endpoint must be a non-empty string or RpcClient"
            )
        if "://" not in endpoint:
            endpoint = f"http://{endpoint}"
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("embedding endpoint must be an HTTP URL")
        return endpoint.rstrip("/")

    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict[str, Any]:
        """Encode one multimodal input, retrying transient transport failures."""
        payload = {"text": text, "image_uris": image_uris or []}
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                if self._rpc is not None:
                    response = self._rpc.call(
                        "embedding.encode",
                        kwargs=payload,
                        timeout_s=self._timeout_s,
                    )
                else:
                    response = self._encode_http(payload)
                return self._validate_response(response)
            except (OSError, TimeoutError, RpcError) as exc:
                last_error = exc
                if attempt == self._retries:
                    break
                if self._retry_backoff_s:
                    time.sleep(self._retry_backoff_s * (2**attempt))
        assert last_error is not None
        raise TimeoutError(
            f"embedding encode failed after {self._retries + 1} attempts: {last_error}"
        ) from last_error

    def _encode_http(self, payload: dict[str, Any]) -> Any:
        assert self._endpoint is not None
        request = urllib.request.Request(
            f"{self._endpoint}/encode",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            open_request = self._opener.open if self._opener else urllib.request.urlopen
            with open_request(request, timeout=self._timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                message = json.loads(raw).get("error", str(exc))
            except (AttributeError, json.JSONDecodeError):
                message = str(exc)
            if 400 <= exc.code < 500:
                raise ValueError(f"embedding encode rejected: {message}") from exc
            raise RpcError("embedding.encode", message) from exc
        except OSError as exc:
            raise RpcError("embedding.encode", f"HTTP request failed: {exc}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RpcError("embedding.encode", f"invalid JSON response: {exc}") from exc

    @staticmethod
    def _validate_response(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("embedding response must be an object")
        vector = payload.get("vector")
        dim = payload.get("dim")
        model = payload.get("model")
        if (
            not isinstance(vector, list)
            or not vector
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            )
        ):
            raise RuntimeError("embedding response contains an invalid vector")
        if type(dim) is not int or dim != len(vector):
            raise RuntimeError("embedding response dim does not match vector length")
        if model != EMBEDDING_MODEL_ID:
            raise RuntimeError(
                f"unexpected embedding model {model!r}; expected {EMBEDDING_MODEL_ID!r}"
            )
        return {
            "vector": [float(value) for value in vector],
            "model": model,
            "dim": dim,
        }
