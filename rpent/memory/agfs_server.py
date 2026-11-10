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

"""Parent-death-watching launcher for the external agfs-server binary."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading

from rpent.utils.daemon import watch_parent_death


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", default="agfs-server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--parent-watch", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    shutdown = threading.Event()
    try:
        process = subprocess.Popen(
            [args.server_bin, "-c", args.config, "-addr", args.address],
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"failed to launch agfs-server: {exc}", file=sys.stderr, flush=True)
        return 1

    def request_shutdown(*_: object) -> None:
        shutdown.set()

    watcher = watch_parent_death(request_shutdown) if args.parent_watch else None
    previous_handlers = {}
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_number] = signal.signal(
            signal_number, request_shutdown
        )
    try:
        while not shutdown.wait(0.1):
            returncode = process.poll()
            if returncode is not None:
                return returncode
        process.terminate()
        try:
            return process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if watcher is not None:
            watcher.close()
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
