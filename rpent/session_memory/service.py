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

"""Independent mem0 service for operator-scoped session and preference memory."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade, make_rpc_client

logger = get_logger("session_memory")

SESSION_MEMORY_COLLECTION = "session_mem__prod"
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_OPERATOR_ID = "default_operator"


def build_mem0_config(
    *,
    milvus_uri: str = DEFAULT_MILVUS_URI,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OSS mem0 configuration pinned to the session Milvus collection."""
    if not isinstance(milvus_uri, str) or not milvus_uri.strip():
        raise ValueError("milvus_uri must be a non-empty string")
    resolved = copy.deepcopy(config) if config is not None else {}
    if not isinstance(resolved, dict):
        raise ValueError("mem0 config must be a JSON object")
    if resolved.get("graph"):
        raise ValueError("mem0 graph memory is not supported")
    vector_store = resolved.setdefault("vector_store", {})
    if not isinstance(vector_store, dict):
        raise ValueError("mem0 vector_store config must be an object")
    vector_config = vector_store.setdefault("config", {})
    if not isinstance(vector_config, dict):
        raise ValueError("mem0 vector_store.config must be an object")
    vector_store["provider"] = "milvus"
    vector_config.update(
        {
            "collection_name": SESSION_MEMORY_COLLECTION,
            "embedding_model_dims": 1536,
            "url": milvus_uri.strip(),
        }
    )
    return resolved


def _scope(user_id: str, run_id: str) -> tuple[str, str]:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    return user_id, run_id


class SessionMemoryService:
    """Own the mem0 SDK inside the service process, never in the agent process."""

    def __init__(
        self,
        *,
        milvus_uri: str = DEFAULT_MILVUS_URI,
        config: dict[str, Any] | None = None,
        read_only: bool = False,
        memory: Any = None,
    ) -> None:
        self.read_only = bool(read_only)
        if memory is not None:
            self._memory = memory
            return
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise RuntimeError(
                "mem0 dependencies are missing; install RPent with "
                '`pip install -e ".[session-memory]"`'
            ) from exc
        try:
            self._memory = Memory.from_config(
                build_mem0_config(milvus_uri=milvus_uri, config=config)
            )
        except Exception as exc:
            raise RuntimeError(f"failed to initialize mem0: {exc}") from exc

    def add(
        self,
        *,
        content: str,
        event: str,
        user_id: str,
        run_id: str,
    ) -> Any:
        """Add one operator interaction under both required scopes."""
        if self.read_only:
            raise PermissionError("session memory is read-only")
        user_id, run_id = _scope(user_id, run_id)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if event not in {"task", "user_input", "verdict"}:
            raise ValueError(f"unsupported session memory event: {event!r}")
        return self._memory.add(
            [{"role": "user", "content": content}],
            user_id=user_id,
            run_id=run_id,
            metadata={"event": event},
        )

    def search(
        self,
        *,
        query: str,
        user_id: str,
        run_id: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search only the supplied operator and cell scopes."""
        user_id, run_id = _scope(user_id, run_id)
        if not isinstance(query, str) or not query.strip():
            return {"results": []}
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        result = self._memory.search(
            query,
            user_id=user_id,
            run_id=run_id,
            limit=top_k,
        )
        results = result.get("results", []) if isinstance(result, dict) else result
        if not isinstance(results, list):
            raise RuntimeError("mem0 search returned an invalid result")
        return {"results": results}


class SessionMemoryFacade(RpcFacade):
    """Expose the session service over the repository's HTTP RPC transport."""

    def __init__(self, **service_kwargs: Any) -> None:
        super().__init__()
        self.service = SessionMemoryService(**service_kwargs)
        self._rpc["session_memory.add"] = self.service.add
        self._rpc["session_memory.search"] = self.service.search
        self._readonly_methods.add("session_memory.search")


class SessionMemoryClient:
    """Thin RPC client; importing it never imports mem0 or model code."""

    def __init__(self, endpoint: str, *, timeout_s: float = 5.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._rpc = make_rpc_client(endpoint)
        self._timeout_s = float(timeout_s)

    def add(self, **kwargs: Any) -> Any:
        return self._rpc.call(
            "session_memory.add", kwargs=kwargs, timeout_s=self._timeout_s
        )

    def search(self, **kwargs: Any) -> dict[str, Any]:
        return self._rpc.call(
            "session_memory.search", kwargs=kwargs, timeout_s=self._timeout_s
        )


def create_session_memory_client(
    endpoint: str | None, *, timeout_s: float = 5.0
) -> SessionMemoryClient | None:
    """Create an optional client, degrading to disabled session memory on error."""
    endpoint = endpoint or os.environ.get("SESSION_MEMORY_ENDPOINT")
    if not endpoint:
        return None
    try:
        return SessionMemoryClient(endpoint, timeout_s=timeout_s)
    except Exception as exc:
        logger.warning("session memory disabled: %s", exc)
        return None


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RPent mem0 session memory server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--milvus-uri",
        default=os.environ.get("SESSION_MEMORY_MILVUS_URI", DEFAULT_MILVUS_URI),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON mem0 OSS config for local LLM/embedder providers. Milvus settings are enforced.",
    )
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--parent-watch", action="store_true")
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    try:
        config = json.loads(args.config.read_text()) if args.config else None
        facade = SessionMemoryFacade(
            milvus_uri=args.milvus_uri,
            config=config,
            read_only=args.read_only,
        )
    except Exception as exc:
        logger.warning("session memory service unavailable: %s", exc)
        return 1
    facade.serve(
        transport="http",
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
