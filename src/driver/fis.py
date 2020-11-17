from argparse import ArgumentParser, ArgumentTypeError, FileType
from typing import Tuple


def argStringParser(string: str) -> Tuple[str, float]:
    if not ":" in string:
        raise ArgumentTypeError(
            f"Bad argument {string}. It should be a string in the form <VariableName:Value>"
        )
    name, value = string.split(sep=":")
    value = float(value)
    return name, value


if __name__ == "__main__":
    argParser = ArgumentParser(
        description="Tool for managing Mamdani and Larsen Fuzzy Inference Systems"
    )

    argParser.add_argument(
        "-l",
        "--load",
        type=FileType("r"),
        dest="file",
        help="File containing the FIS definition with the specified grammar.",
    )

    argParser.add_argument(
        "-s",
        "--show",
        dest="show",
        action="store_const",
        const=True,
        default=False,
        help="Show the Domain Specific Language available for the construction of FIS.",
    )

    argParser.add_argument(
        "-n",
        "--no-interactive",
        dest="interactive",
        action="store_const",
        default=False,
        const=True,
        help="""Instead of stay in an interactive session, it ends the loaded system. Note
                that this option should be used in conjuction with --load and --eval.
        """,
    )

    argParser.add_argument(
        "-e",
        "--eval",
        dest="inputVariables",
        type=argStringParser,
        help="Provides inputVariables to the loaded system. Should be used in conjuction with --load.",
        nargs="+",
    )
