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

"""Rebuild one Milvus episode collection from canonical AGFS JSON files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rpent.episode.milvus_sink import (
    episode_collection_name,
    rebuild_episode_index,
)
from rpent.robots.components.embedding_client import EmbeddingClient
from rpent.robots.components.embedding_server import EmbeddingFacade


class LocalMockEmbedding:
    def __init__(self, memory_root: Path) -> None:
        self._facade = EmbeddingFacade(mock=True)
        self._frames_dir = (
            memory_root / "_internal" / "agfs" / "frames"
        ).resolve()

    def encode(
        self, text: str = "", image_uris: list[str] | None = None
    ) -> dict:
        local_uris = []
        for uri in image_uris or []:
            if not uri.startswith("agfs://frames/"):
                raise ValueError(f"local mock cannot resolve image URI: {uri}")
            path = (self._frames_dir / uri.removeprefix("agfs://frames/")).resolve()
            if not path.is_relative_to(self._frames_dir):
                raise ValueError(f"frame URI escapes local storage: {uri}")
            local_uris.append(path.as_uri())
        return self._facade.encode(text=text, image_uris=local_uris)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--memory-version", required=True)
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path("memory"),
        help="Parent of robot memory directories (default: memory).",
    )
    parser.add_argument(
        "--milvus-uri",
        default=None,
        help="Standalone URI or local DB path (default: the robot's .milvus/episodes.db).",
    )
    embedding = parser.add_mutually_exclusive_group(required=True)
    embedding.add_argument("--embedding-endpoint", default=None)
    embedding.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic local mock encoder for CI and regression checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    memory_root = (args.memory_dir / args.robot).resolve()
    episodes_dir = memory_root / "_internal" / "agfs" / "episodes"
    embedding_client = (
        LocalMockEmbedding(memory_root)
        if args.mock
        else EmbeddingClient(args.embedding_endpoint)
    )
    count = rebuild_episode_index(
        episodes_dir=episodes_dir,
        memory_version=args.memory_version,
        embedding_client=embedding_client,
        memory_root=memory_root,
        milvus_uri=args.milvus_uri,
    )
    print(
        f"Rebuilt {count} episodes in "
        f"{episode_collection_name(args.memory_version)} at "
        f"{args.milvus_uri or memory_root / '.milvus' / 'episodes.db'}"
    )


if __name__ == "__main__":
    main()
