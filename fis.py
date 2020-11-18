#! /usr/bin/python

from argparse import ArgumentParser, ArgumentTypeError, FileType
from src.fuzzyevaluator.evaluate import (
    FuzzySystem,
    LarsenFuzzySystem,
    MamdaniFuzzySystem,
)
from typing import Tuple
from src.fuzzydsl.dsl import Parser


def argStringParser(string: str) -> Tuple[str, float]:
    if not ":" in string:
        raise ArgumentTypeError(
            f"Bad argument {string}. It should be a string in the form <VariableName:Value>"
        )
    name, value = string.split(sep=":")
    try:
        value = float(value)
    except Exception as e:
        raise ArgumentTypeError(
            f"Bad argument {string}. It should be a string in the form <VariableName:Value>"
        )
    return name, value


modelTable = {"mamdani": MamdaniFuzzySystem, "larsen": LarsenFuzzySystem}


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
        "-r",
        "--read-input",
        dest="input",
        type=FileType("r"),
        help="""
            Read input variables from a file. This options requires --load. 
        """,
    )

    argParser.add_argument(
        "-e",
        "--eval",
        dest="inputVariables",
        type=argStringParser,
        help="Provides inputVariables to the loaded system. Should be used in conjuction with --load.",
        nargs="+",
        metavar="<VariableName:Value>",
    )

    argParser.add_argument(
        "-m",
        "--method",
        dest="method",
        choices=["maximum", "centroid", "mom", "bisector"],
        help="Defuzzification method used by the system. Defaults to centroid.",
        default="centroid",
    )

    argParser.add_argument(
        "-t",
        "--type",
        choices=["mamdani", "larsen"],
        default="mamdani",
        help="Define the model to use in the system. Defaults to mamdani.",
    )

    args = argParser.parse_args()

    if args.inputVariables:
        # We are trying to eval an input
        # Check that there is a loaded System
        if not args.file:
            print(argParser.usage)
            print(
                "--eval option must be used when a system is loaded, use option --file"
            )
            exit()

        # Load the system
        parser = Parser()
        # Parse the system from file
        sets, rules = parser(args.file.read())
        # build the system
        fuzzyInferenceSystem: FuzzySystem = modelTable[args.type](
            sets, rules, args.method
        )
        # Evaluate the provided input
        # First build the input as a dict
        input_ = {var: value for var, value in args.inputVariables}
        print(fuzzyInferenceSystem.evaluate(input_))
    elif args.input:
        # We are trying to eval an input
        # Check that there is a loaded System
        if not args.file:
            print(argParser.usage)
            print(
                "--eval option must be used when a system is loaded, use option --file"
            )
            exit()

        # Load the system
        parser = Parser()
        # Parse the system from file
        sets, rules = parser(args.file.read())
        # build the system
        fuzzyInferenceSystem: FuzzySystem = modelTable[args.type](
            sets, rules, args.method
        )
        # Evaluate the provided input
        # Each file line contains an input
        for line in args.input.readlines():
            inputVar = map(argStringParser, line.split())
            input_ = {var: value for var, value in inputVar}
            print(fuzzyInferenceSystem.evaluate(input_))
    else:
        print("Must suply either -r or -e options")