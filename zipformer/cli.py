#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.            (authors: Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import logging


def main():
    commands = {
        "train": "zipformer.bin.train",
        "decode": "zipformer.bin.decode",
        "export": "zipformer.bin.export",
        "inference": "zipformer.bin.inference",
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        usage = "usage: zipformer <command> [<args>]\n\nAvailable commands:\n"
        for cmd in commands:
            usage += f"  {cmd}\n"
        print(usage)
        sys.exit(0)

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    command = sys.argv[1]
    if command not in commands:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(commands)}")
        sys.exit(1)

    # Remove 'zipformer' and the subcommand from argv so the
    # subcommand's argparse sees only its own arguments.
    sys.argv = [f"zipformer {command}"] + sys.argv[2:]

    from importlib import import_module

    module = import_module(commands[command])
    module.main()
