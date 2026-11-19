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

"""Freeze a memory version for reproducible evaluation."""

import logging

from rpent.memory.freeze import freeze_memory_version, freeze_parser, logger


def main(argv: list[str] | None = None) -> int:
    args = freeze_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    try:
        manifest = freeze_memory_version(**vars(args))
    except Exception as exc:
        logger.error("Freeze failed: %s: %s", type(exc).__name__, exc)
        return 1
    logger.info("Frozen memory manifest: %s", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
