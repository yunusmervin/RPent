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

"""Check the canonical corpus, Milvus properties and AGFS freeze policy."""

import logging

from rpent.memory.freeze import check_freeze, freeze_parser, logger


def main(argv: list[str] | None = None) -> int:
    args = freeze_parser(check=True).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    errors = check_freeze(**vars(args))
    for error in errors:
        logger.error("Freeze check failed: %s", error)
    if errors:
        return 1
    logger.info("Freeze check passed: %s/%s", args.robot, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
