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

"""Freeze canonical memory and verify its derived index without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from rpent.episode.milvus_sink import episode_collection_name
from rpent.episode.model import Episode
from rpent.memory.freeze_state import freeze_mounts, read_freeze_config
from rpent.memory.source import parse_repo
from rpent.utils.logging import get_logger

UTC = timezone.utc
logger = get_logger("memory.freeze")
_EXCLUDED = {"_internal", ".milvus", ".git"}
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def _component(value: str) -> str:
    if not _COMPONENT.fullmatch(value):
        raise ValueError(f"invalid robot or version directory name: {value!r}")
    return value


def robot_root(memory_dir: str | Path, robot: str) -> Path:
    """Resolve a single robot subtree, rejecting symlink escapes."""
    parent = Path(memory_dir).resolve()
    root = parent / _component(robot)
    if root.is_symlink() or not root.resolve().is_relative_to(parent):
        raise ValueError("robot memory root must not escape its parent")
    return root


def _inside(root: Path, path: str | Path) -> Path:
    candidate = Path(path).absolute()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or candidate != resolved:
        raise ValueError(f"memory path escapes robot root or uses a symlink: {path}")
    return resolved


def _snapshot_paths(directory: Path, *, corpus: bool = False) -> list[Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(
            f"snapshot directory does not exist or is a symlink: {directory}"
        )
    result = [directory]

    def walk(parent: Path) -> None:
        for path in sorted(parent.iterdir()):
            if (
                corpus
                and parent == directory
                and (path.name in _EXCLUDED or path.name.startswith("."))
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"snapshot must not contain symlinks: {path}")
            result.append(path)
            if path.is_dir():
                walk(path)
            elif not path.is_file():
                raise ValueError(f"unsupported snapshot entry: {path}")

    walk(directory)
    return result


def _inventory(directory: Path, *, corpus: bool = False) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in _snapshot_paths(directory, corpus=corpus)
        if path.is_file()
    }


def _git(directory: Path, *args: str) -> str:
    return (
        subprocess.check_output(
            ["git", "-C", str(directory), *args], stderr=subprocess.PIPE
        )
        .decode()
        .strip()
    )


def _verify_git(
    corpus: Path, files: dict[str, str], revision: str, repository: Path
) -> None:
    if not _COMMIT.fullmatch(revision):
        raise ValueError("corpus revision must be an immutable git commit")
    if _git(repository, "cat-file", "-t", revision) != "commit":
        raise ValueError("corpus revision is not a git commit")
    prefix = corpus.relative_to(repository)
    tree = subprocess.check_output(
        ["git", "-C", str(repository), "ls-tree", "-r", "-z", revision],
        stderr=subprocess.PIPE,
    )
    blobs = {}
    for entry in tree.split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, kind, object_id = metadata.decode().split()
            if kind == "blob" and mode != "120000":
                blobs[os.fsdecode(name)] = object_id
    algorithm = _git(repository, "rev-parse", "--show-object-format")
    committed = set()
    for name in blobs:
        path = Path(name)
        if path.is_relative_to(prefix):
            relative = path.relative_to(prefix)
            if (
                relative.parts
                and relative.parts[0] not in _EXCLUDED
                and not relative.parts[0].startswith(".")
            ):
                committed.add(relative.as_posix())
    if committed != set(files):
        raise ValueError("corpus file list differs from the committed revision")
    for relative in files:
        content = (corpus / relative).read_bytes()
        blob = f"blob {len(content)}\0".encode() + content
        digest = hashlib.new(algorithm, blob, usedforsecurity=False).hexdigest()
        if blobs.get((prefix / relative).as_posix()) != digest:
            raise ValueError(
                f"corpus file is uncommitted or differs from {revision}: {relative}"
            )


def _provenance(
    corpus: Path,
    files: dict[str, str],
    robot: str,
    version: str,
    source: str | None,
    root: Path,
) -> dict[str, str]:
    if source is None:
        repository = Path(_git(corpus, "rev-parse", "--show-toplevel"))
        revision = _git(repository, "rev-parse", "HEAD")
        _verify_git(corpus, files, revision, repository)
        return {
            "corpus_revision": f"git:{revision}",
            "corpus_git_root": str(repository),
        }

    from huggingface_hub import HfApi, hf_hub_download

    repo, requested = parse_repo(source)
    info = HfApi().repo_info(repo, repo_type="dataset", revision=requested)
    revision = info.sha
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        raise ValueError("HF did not resolve an immutable corpus commit")
    prefix = f"{robot}/{version}/" if corpus == root / version else f"{robot}/"
    remote = {
        entry.rfilename.removeprefix(prefix)
        for entry in info.siblings or []
        if entry.rfilename.startswith(prefix)
        and not entry.rfilename.removeprefix(prefix).split("/")[0].startswith(".")
        and entry.rfilename.removeprefix(prefix).split("/")[0] not in _EXCLUDED
    }
    if remote != set(files):
        raise ValueError("local corpus file list differs from the pinned HF corpus")
    for relative, digest in files.items():
        downloaded = Path(
            hf_hub_download(
                repo, prefix + relative, repo_type="dataset", revision=revision
            )
        )
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != digest:
            raise ValueError(
                f"local corpus differs from pinned HF revision: {relative}"
            )
    return {"corpus_revision": f"hf:{repo}@{revision}", "corpus_prefix": prefix}


def _readonly(directory: Path, *, corpus: bool = False) -> None:
    paths = _snapshot_paths(directory, corpus=corpus)
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".freeze-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _milvus_rows(
    client: MilvusClient, collection: str, robot: str
) -> list[dict[str, Any]]:
    client.load_collection(collection_name=collection)
    iterator = client.query_iterator(
        collection_name=collection,
        filter=f"robot == {json.dumps(robot)}",
        output_fields=["*"],
        batch_size=1000,
    )
    rows = []
    try:
        while batch := iterator.next():
            for row in batch:
                row["vector"] = [float(value) for value in row["vector"]]
            rows.extend(batch)
    finally:
        iterator.close()
    return sorted(rows, key=lambda row: row["episode_id"])


def _row_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _verify_episode_rows(rows: list[dict[str, Any]], episodes: list[Episode]) -> None:
    fields = (
        "episode_id",
        "robot",
        "suite",
        "regime",
        "task_id",
        "seed",
        "outcome",
        "confidence",
        "memory_ids",
    )
    expected = {}
    for episode in episodes:
        row = {field: getattr(episode, field) for field in fields}
        row["episode_uri"] = (
            f"agfs://episodes/{episode.memory_version}/{episode.episode_id}.json"
        )
        expected[episode.episode_id] = row
    actual = {
        row["episode_id"]: {
            field: value for field, value in row.items() if field != "vector"
        }
        for row in rows
    }
    if actual != expected:
        raise ValueError(
            "Milvus rows differ from canonical episodes for this robot/version"
        )


def freeze_memory_version(
    *,
    robot: str,
    version: str,
    memory_dir: str | Path = "memory",
    corpus_dir: str | Path | None = None,
    memory_repo: str | None = None,
    milvus_uri: str | None = None,
) -> Path:
    """Freeze one existing corpus and index; never modify canonical content.

    Writers and AGFS servers must be stopped before this administrative operation.
    A repeated call verifies the existing snapshot instead of replacing it.
    """
    _component(version)
    root = robot_root(memory_dir, robot)
    manifest_path = _inside(
        root, root / "_internal" / "freeze" / f"{version}.manifest.json"
    )
    if manifest_path.exists():
        errors = check_freeze(
            robot=robot, version=version, memory_dir=memory_dir, milvus_uri=milvus_uri
        )
        if errors:
            raise ValueError("existing freeze is inconsistent: " + "; ".join(errors))
        return manifest_path
    default = root / version if (root / version).is_dir() else root
    corpus = _inside(root, corpus_dir if corpus_dir is not None else default)
    episodes = _inside(root, root / "_internal" / "agfs" / "episodes" / version)
    files = _inventory(corpus, corpus=True)
    if not files:
        raise ValueError("cannot freeze an empty corpus")
    provenance = _provenance(corpus, files, robot, version, memory_repo, root)
    episode_files = _inventory(episodes)
    frames = {}
    canonical_episodes = []
    for relative in episode_files:
        path = episodes / relative
        episode = Episode.model_validate_json(path.read_bytes())
        if (
            episode.robot != robot
            or episode.memory_version != version
            or path.stem != episode.episode_id
        ):
            raise ValueError(f"episode does not belong to this robot/version: {path}")
        canonical_episodes.append(episode)
        for uri in (
            episode.assets.first_frame,
            episode.assets.success_frame,
            *episode.assets.failure_frames,
            *episode.assets.seg_overlays,
        ):
            if uri is None:
                continue
            frame = _inside(
                root,
                root
                / "_internal"
                / "agfs"
                / "frames"
                / uri.removeprefix("agfs://frames/"),
            )
            content_hash = hashlib.sha256(frame.read_bytes()).hexdigest()
            if frame.stem != content_hash:
                raise ValueError(f"frame content does not match its address: {frame}")
            frames[frame.relative_to(root).as_posix()] = content_hash
    uri = milvus_uri or str(root / ".milvus" / "episodes.db")
    if "://" not in uri:
        uri = str(_inside(root, uri))
    _require_database(uri)
    collection = episode_collection_name(version)
    client = MilvusClient(uri=uri)
    try:
        if not client.has_collection(collection_name=collection):
            raise ValueError(f"Milvus collection does not exist: {collection}")
        properties = (
            client.describe_collection(collection_name=collection).get("properties")
            or {}
        )
        model = properties.get("embedding_model")
        if not isinstance(model, str) or not model:
            raise ValueError("Milvus collection has no embedding_model property")
        for relative in episode_files:
            if (
                Episode.model_validate_json(
                    (episodes / relative).read_bytes()
                ).embedding_model
                != model
            ):
                raise ValueError("episode and collection embedding models differ")
        client.flush(collection_name=collection)
        rows = _milvus_rows(client, collection, robot)
        _verify_episode_rows(rows, canonical_episodes)
        row_digest = _row_digest(rows)
        frozen_at = properties.get("frozen_at") or datetime.now(
            UTC
        ).isoformat().replace("+00:00", "Z")
        if datetime.fromisoformat(
            frozen_at.replace("Z", "+00:00")
        ).utcoffset() != timedelta(0):
            raise ValueError("Milvus frozen_at must be a UTC timestamp")
        client.alter_collection_properties(
            collection_name=collection,
            properties={"frozen": "true", "frozen_at": frozen_at},
        )
        client.release_collection(collection_name=collection)
        row_count = client.get_collection_stats(collection_name=collection)["row_count"]
    finally:
        client.close()
    # Persist backend policy before removing write bits or publishing the receipt.
    freeze_mounts(root, version, corpus, episodes, frozen_at)
    _readonly(corpus, corpus=True)
    _readonly(episodes)
    for relative in frames:
        frame = root / relative
        frame.chmod(stat.S_IMODE(frame.stat().st_mode) & ~0o222)
    manifest = {
        "version": version,
        "robot": robot,
        **provenance,
        "milvus_collection": collection,
        "milvus_uri": uri,
        "milvus_row_count": row_count,
        "milvus_rows_sha256": row_digest,
        "embedding_model": model,
        "frozen_at": frozen_at,
        "corpus_root": str(corpus),
        "episodes_root": str(episodes),
        "corpus_sha256": files,
        "episodes_sha256": episode_files,
        "frames_sha256": frames,
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _require_database(uri: str) -> None:
    if "://" not in uri and not Path(uri).exists():
        raise ValueError(f"Milvus database does not exist: {uri}")


def check_freeze(
    *,
    robot: str,
    version: str,
    memory_dir: str | Path = "memory",
    milvus_uri: str | None = None,
) -> list[str]:
    """Return inconsistencies; an empty result means all snapshot checks passed."""
    try:
        _component(version)
        root = robot_root(memory_dir, robot)
        path = _inside(root, root / "_internal" / "freeze" / f"{version}.manifest.json")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["version"] != version or manifest["robot"] != robot:
            raise ValueError("manifest robot/version mismatch")
        collection = episode_collection_name(version)
        if manifest["milvus_collection"] != collection:
            raise ValueError("manifest collection name mismatch")
        frozen_at = manifest["frozen_at"]
        if datetime.fromisoformat(
            frozen_at.replace("Z", "+00:00")
        ).utcoffset() != timedelta(0):
            raise ValueError("manifest frozen_at must be a UTC timestamp")
        corpus = _inside(root, manifest["corpus_root"])
        episodes = _inside(root, manifest["episodes_root"])
        if episodes != root / "_internal" / "agfs" / "episodes" / version:
            raise ValueError("manifest episode path mismatch")
        errors = []
        for directory, field, is_corpus in (
            (corpus, "corpus_sha256", True),
            (episodes, "episodes_sha256", False),
        ):
            actual = _inventory(directory, corpus=is_corpus)
            if actual != manifest[field]:
                errors.append(f"{field}: snapshot content differs")
            for path in _snapshot_paths(directory, corpus=is_corpus):
                if path.stat().st_mode & 0o222:
                    errors.append(f"snapshot path is writable: {path}")
        for relative, digest in manifest["frames_sha256"].items():
            frame = _inside(root, root / relative)
            if (
                hashlib.sha256(frame.read_bytes()).hexdigest() != digest
                or frame.stat().st_mode & 0o222
            ):
                errors.append(
                    f"frame content or readonly permission differs: {relative}"
                )
        revision = manifest["corpus_revision"]
        if revision.startswith("git:"):
            _verify_git(
                corpus,
                manifest["corpus_sha256"],
                revision.removeprefix("git:"),
                Path(manifest["corpus_git_root"]),
            )
        elif revision.startswith("hf:"):
            _, separator, commit = revision.rpartition("@")
            if not separator or not _COMMIT.fullmatch(commit):
                raise ValueError("manifest HF revision must be an immutable commit")
            parse_repo(revision.removeprefix("hf:"))
        else:
            raise ValueError("manifest has no supported corpus revision")
        entry = read_freeze_config(root)["versions"].get(version)
        expected = {
            "corpus_root": str(corpus),
            "episodes_dir": str(episodes),
            "frozen_at": frozen_at,
        }
        if entry != expected:
            errors.append("AGFS readonly configuration differs from manifest")
        uri = milvus_uri or manifest["milvus_uri"]
        if "://" not in uri:
            uri = str(_inside(root, uri))
        if uri != manifest["milvus_uri"]:
            errors.append("Milvus URI differs from manifest")
        _require_database(uri)
        client = MilvusClient(uri=uri)
        try:
            if not client.has_collection(collection_name=collection):
                errors.append(f"Milvus collection does not exist: {collection}")
            else:
                properties = (
                    client.describe_collection(collection_name=collection).get(
                        "properties"
                    )
                    or {}
                )
                if (
                    str(properties.get("frozen")).lower() != "true"
                    or properties.get("frozen_at") != frozen_at
                ):
                    errors.append("Milvus frozen properties differ from manifest")
                if properties.get("embedding_model") != manifest["embedding_model"]:
                    errors.append("Milvus embedding model differs from manifest")
                if (
                    client.get_collection_stats(collection_name=collection)["row_count"]
                    != manifest["milvus_row_count"]
                ):
                    errors.append("Milvus row count differs from manifest")
                if (
                    _row_digest(_milvus_rows(client, collection, robot))
                    != manifest["milvus_rows_sha256"]
                ):
                    errors.append("Milvus row content differs from manifest")
                client.release_collection(collection_name=collection)
        finally:
            client.close()
        return list(dict.fromkeys(errors))
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"]


def freeze_parser(*, check: bool = False) -> argparse.ArgumentParser:
    """Build the shared administrative CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Verify a frozen memory snapshot."
        if check
        else "Freeze one corpus, Milvus collection and AGFS episode version."
    )
    parser.add_argument("--robot", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path("memory"),
        help="Parent of robot memory directories.",
    )
    parser.add_argument("--milvus-uri", default=None)
    if not check:
        parser.add_argument(
            "--corpus-dir",
            type=Path,
            default=None,
            help="Corpus inside this robot root; defaults to <robot>/<version> if present, otherwise <robot>.",
        )
        parser.add_argument(
            "--memory-repo",
            default=None,
            help="HF dataset owner/repo@revision; otherwise verify the local corpus git commit.",
        )
    return parser
