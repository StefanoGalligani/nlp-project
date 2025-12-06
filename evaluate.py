# !/usr/bin/env python3
# conding=utf-8

import os
import sys
import json
import logging
from mwzeval.metrics import Evaluator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bleu",
        dest="bleu",
        action="store_true",
        default=False,
        help="If set, BLEU is evaluated.",
    )
    parser.add_argument(
        "-d",
        "--dst",
        dest="dst",
        action="store_true",
        default=False,
        help="If set, dst is evaluated.",
    )
    parser.add_argument(
        "-s",
        "--success",
        dest="success",
        action="store_true",
        default=False,
        help="If set, inform and success rates are evaluated.",
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input JSON file path."
    )
    parser.add_argument(
        "-r",
        "--richness",
        dest="richness",
        action="store_true",
        default=False,
        help="If set, various lexical richness metrics are evaluated.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file path, here will be the final report.",
    )
    parser.add_argument(
        "-g",
        "--golden",
        type=str,
        default="default",
        help="Golden file to score against.",
    )
    parser.add_argument(
        "--do_not_norm", action="store_true", help="do not normalize states"
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default="",
        help="Log file name. Defaults to outtput with .log as extension.",
    )

    args = parser.parse_args()

    args.norm = not args.do_not_norm

    args.input = os.path.realpath(args.input)
    args.output = os.path.realpath(args.output)
    args.golden = os.path.realpath(args.golden)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not args.log:
        if args.output.endswith(".json"):
            args.log = args.output.replace(".json", ".log")
        else:
            args.log = args.output + ".log"

    if not args.bleu and not args.success and not args.richness and not args.dst:
        sys.stderr.write(
            "error: Missing argument, at least one of -b, -d, -s, and -r must be used!\n"
        )
        parser.print_help()
        sys.exit(1)

    logging.root.setLevel(logging.INFO)
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(filename)s:%(lineno)d - %(message)s",
        filemode="w",
    )
    logger = logging.getLogger(__name__)
    print("- Log file", args.log)
    logger.info("Arguments: %s", args)

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    e = Evaluator(
        args.bleu,
        args.success,
        args.richness,
        dst=args.dst,
        golden=args.golden,
        normalize=args.norm,
    )
    results = e.evaluate(input_data)

    for metric, values in results.items():
        if values is not None:
            print(f"====== {metric.upper()} ======")
            for k, v in values.items():
                print(f"{k.ljust(16)}{v:.2f}")
    print("=" * 24)

    results["args"] = args.__dict__
    with open(args.output, "w+", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("- Results saved to", args.output)
