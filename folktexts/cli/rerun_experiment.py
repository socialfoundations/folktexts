#!/usr/bin/env python3
"""Helper script to re-run a single experiment locally.
"""
from argparse import ArgumentParser
from subprocess import call

from folktexts._io import load_json

from .experiments import Experiment


def setup_arg_parser() -> ArgumentParser:
    # Init parser
    parser = ArgumentParser(description="Re-run a single experiment from its JSON config file.")
    parser.add_argument(
        "--experiment-json",
        type=str,
        help="[string] Path to an experiment JSON file to load.",
        required=True,
    )
    # TODO: add over-writable key-word arguments

    return parser


if __name__ == '__main__':

    # Parse command-line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Load experiment from JSON file
    print(f"Running experiment from config file at '{args.experiment_json}'...")
    exp = Experiment(**load_json(args.experiment_json))

    # Run the experiment
    cmdline_args = [
        cmd
        for key, value in exp.kwargs.items()
        for cmd in (f"--{key.replace('_', '-')}", str(value))
    ]

    call([exp.executable_path] + cmdline_args)
