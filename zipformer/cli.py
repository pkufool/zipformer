import sys


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
