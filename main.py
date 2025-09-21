import argparse
import sys
from pathlib import Path
from src.entry import entry_point
from src.logger import logger


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "-i", "--inputDir", default=["inputs"], nargs="*", type=str, dest="input_paths",
        help="Specify an input directory."
    )
    argparser.add_argument(
        "-t", "--template", required=False, dest="template_file",
        help="Specify a template JSON file."
    )
    argparser.add_argument(
        "-d", "--debug", action="store_false", dest="debug",
        help="Enables debugging mode for detailed errors"
    )
    argparser.add_argument(
        "-o", "--outputDir", default="outputs", dest="output_dir",
        help="Specify an output directory."
    )
    argparser.add_argument(
        "-a", "--autoAlign", action="store_true", dest="autoAlign",
        help="Enables automatic template alignment."
    )
    argparser.add_argument(
        "-l", "--setLayout", action="store_true", dest="setLayout",
        help="Set up OMR template layout."
    )

    args, unknown = argparser.parse_known_args()
    args = vars(args)

    if len(unknown) > 0:
        logger.warning(f"\nError: Unknown arguments: {unknown}", unknown)
        argparser.print_help()
        exit(11)
    return args


def entry_point_for_args(args):
    if args["debug"]:
        sys.tracebacklimit = 0

    for root in args["input_paths"]:
        if args["template_file"]:
            # Open template in binary mode
            with open(args["template_file"], "rb") as f:
                entry_point(Path(root), args, template_file=f)
        else:
            entry_point(Path(root), args)


if __name__ == "__main__":
    args = parse_args()
    entry_point_for_args(args)
