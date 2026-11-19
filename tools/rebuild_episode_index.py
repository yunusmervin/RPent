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

"""Rebuild Milvus episode or knowledge collections from canonical files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rpent.episode.milvus_sink import (
    default_milvus_uri,
    episode_collection_name,
    rebuild_episode_index,
)
from rpent.memory.freeze_state import assert_corpus_writable
from rpent.memory.knowledge import MilvusKnowledgeSink
from rpent.robots.components.embedding_client import EmbeddingClient
from rpent.robots.components.embedding_server import EmbeddingFacade


class LocalMockEmbedding:
    def __init__(self, memory_root: Path) -> None:
        self._facade = EmbeddingFacade(mock=True)
        self._frames_dir = (memory_root / "_internal" / "agfs" / "frames").resolve()

    def encode(self, text: str = "", image_uris: list[str] | None = None) -> dict:
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
        "--collection",
        choices=("episodes", "leaves", "all"),
        default="episodes",
        help="Derived collections to rebuild (default: episodes).",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path("memory"),
        help="Parent of robot memory directories (default: memory).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Knowledge corpus inside this robot directory; defaults to <robot>/<version> if present, otherwise <robot>.",
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
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if (
        args.robot in {".", ".."}
        or Path(args.robot).name != args.robot
        or "\\" in args.robot
    ):
        parser.error("--robot must be a single directory name")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    memory_parent = args.memory_dir.expanduser().resolve()
    memory_root = (memory_parent / args.robot).resolve()
    if not memory_root.is_relative_to(memory_parent):
        parser.error("robot directory escapes --memory-dir")
    episodes_dir = memory_root / "_internal" / "agfs" / "episodes"
    embedding_client = (
        LocalMockEmbedding(memory_root)
        if args.mock
        else EmbeddingClient(args.embedding_endpoint)
    )
    if args.collection in {"episodes", "all"}:
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
    if args.collection in {"leaves", "all"}:
        default_corpus = memory_root / args.memory_version
        corpus = (
            args.corpus_dir.expanduser().resolve()
            if args.corpus_dir is not None
            else default_corpus.resolve()
            if default_corpus.is_dir()
            else memory_root
        )
        if not corpus.is_relative_to(memory_root):
            parser.error("knowledge corpus escapes the robot memory directory")
        if not corpus.is_dir():
            parser.error("knowledge corpus directory does not exist")
        assert_corpus_writable(corpus)
        sink = MilvusKnowledgeSink(
            embedding_client,
            memory_root=corpus,
            robot=args.robot,
            memory_version=args.memory_version,
            milvus_uri=args.milvus_uri or default_milvus_uri(memory_root),
        )
        try:
            count = sink.rebuild()
            print(
                f"Rebuilt {count} leaves in {sink.collection_name} at {args.milvus_uri or memory_root / '.milvus' / 'episodes.db'}"
            )
        finally:
            sink.close()


if __name__ == "__main__":
    main()
