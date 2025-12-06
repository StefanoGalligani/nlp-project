#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Preprocesses predictions json and makes sure that it is in the correct format for evaluation.
Removes non-matching keys, or maps to closest ones specific to the dataset.

Uses fuzzy matching to map slot values to the closest ones from the database of names.

Both SA-MultiWOZ and SpokenWOZ ontologies are bundled in mwzeval/data/database/.
Custom ontology paths can be provided via --sa-ontology or --sp-ontology arguments.
"""

import argparse
import sys
import json
import os
from time import time
from fuzzywuzzy import process
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from preprocessor import preprocess_schema, SA_MWOZ, SP_WOZ


# Default paths - use packaged data files
_MWZEVAL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mwzeval")
DEFAULT_SA_DB = os.path.join(
    _MWZEVAL_DIR, "data", "database", "sa_mwoz_train_dev_test_names.json"
)
DEFAULT_SP_DB = os.path.join(
    _MWZEVAL_DIR, "data", "database", "spoken_woz_ontology.json"
)


def load_spokenwoz_domain2names(fname):
    """Load the ontology from the SpokenWoz dataset and convert to domain2 names nested dictionary"""

    ontology = {}
    with open(fname, "r", encoding="utf-8") as fpr:
        ontology = json.load(fpr)

    domain2names = {}
    for dom_slot, names in ontology.items():
        domain, slot = dom_slot.split("-")
        if domain not in domain2names:
            domain2names[domain] = {}
        if slot not in ("time", "leaveat", "arriveby", "people", "stars"):
            domain2names[domain][slot] = names
    return domain2names


def load_sa_mwoz_domain2names(fname):
    """Load domain to slot names from the database json file"""
    domain2names = {}
    with open(fname, "r", encoding="utf-8") as fpr:
        domain2names = json.load(fpr)
    return domain2names


def fuzzy_map_per_dialog(dialog_dict, domain2names: dict):
    """Fuzzy maps the slot values to the closest one from the database of names"""

    new_dialog_dict = {"dialog_id": dialog_dict["dialog_id"], "turn_list": []}
    new_turn_list = []

    for i, turn_dict in enumerate(dialog_dict["turn_list"]):
        new_turn_dict = deepcopy(turn_dict)
        new_turn_list.append(new_turn_dict)

        if turn_dict["state"] is None:
            continue  # Skip this turn if state is None

        for domain, domain_dict in turn_dict["state"].items():
            if domain_dict is None:
                continue  # Skip this domain if domain_dict is None

            for slot_key, slot_val in list(domain_dict.items()):
                if domain in domain2names and slot_key not in (
                    "time",
                    "leaveat",
                    "arriveby",
                    "people",
                    "stars",
                ):
                    # Find closest slot name using fuzzy matching
                    try:
                        closest_name, _ = process.extractOne(
                            slot_val, domain2names[domain][slot_key]
                        )  # type: ignore
                    except TypeError:
                        raise TypeError(
                            f"Error in domain2names[{domain}][{slot_key}] - {domain}: "
                            + json.dumps(slot_val)
                        )
                    # if args.verbose:
                    #    print(f"{i} {domain}:: {slot_key}:{slot_val} -> {closest_name}")

                    new_turn_list[i]["state"][domain][slot_key] = closest_name

    new_dialog_dict["turn_list"] = new_turn_list
    return new_dialog_dict


def main(args):
    """main method"""

    # if args.verbose:
    stime = time()

    if len(sys.argv) > 2:
        data = {}
        with open(args.in_json, "r", encoding="utf-8") as f:
            data = json.load(f)

    else:
        args.verbose = False  # we dont want to print verbose info to stdout
        data = json.load(sys.stdin)

    if args.sa:
        # Speech aware MultiWOZ
        data = preprocess_schema(data, SA_MWOZ, args.verbose)
        sa_db_path = args.sa_ontology if args.sa_ontology else DEFAULT_SA_DB
        if not os.path.exists(sa_db_path):
            raise FileNotFoundError(f"SA-MultiWOZ ontology not found: {sa_db_path}")
        domain2names = load_sa_mwoz_domain2names(sa_db_path)

    if args.sp:
        # Spoken WOZ
        data = preprocess_schema(data, SP_WOZ, args.verbose)
        sp_db_path = args.sp_ontology if args.sp_ontology else DEFAULT_SP_DB
        if not os.path.exists(sp_db_path):
            raise FileNotFoundError(f"SpokenWOZ ontology not found: {sp_db_path}")
        domain2names = load_spokenwoz_domain2names(sp_db_path)

    if args.verbose:
        print("\nDone fixing schema. Now fuzzy mapping slot values.")

    # convert dialog dict to list of dialog dicts for parallel processing
    dialog_list = [
        {"dialog_id": dialog_id, "turn_list": turn_list}
        for dialog_id, turn_list in data.items()
    ]
    fuzzy_map_per_dialog_partial = partial(
        fuzzy_map_per_dialog, domain2names=domain2names
    )

    with Pool(args.nj) as p:
        new_dialog_list = p.map(fuzzy_map_per_dialog_partial, dialog_list)

    # convert list of dialog dicts back to dialog dict
    new_data = {
        dialog_dict["dialog_id"]: dialog_dict["turn_list"]
        for dialog_dict in new_dialog_list
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4)
        if args.verbose:
            print(args.out_json, "saved.")
            print("Time taken:", time() - stime)
    else:
        json.dump(new_data, sys.stdout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    me_group = parser.add_mutually_exclusive_group(required=True)
    me_group.add_argument(
        "--sa", action="store_true", help="Bool flag required for MultiWOZ dataset"
    )
    me_group.add_argument(
        "--sp", action="store_true", help="Bool flag required for SpokenWoz dataset"
    )
    parser.add_argument(
        "--in_json",
        type=str,
        help="Input json file or you can pipe through stdin",
        required=False,
    )
    parser.add_argument(
        "--out_json",
        type=str,
        help="Output json file or it will be on stdout",
        required=False,
    )
    parser.add_argument(
        "--sa-ontology",
        type=str,
        help=f"Path to SA-MultiWOZ ontology JSON (default: {DEFAULT_SA_DB})",
        required=False,
    )
    parser.add_argument(
        "--sp-ontology",
        type=str,
        help=f"Path to SpokenWOZ ontology JSON (default: {DEFAULT_SP_DB})",
        required=False,
    )
    parser.add_argument("--nj", type=int, default=4, help="Number of parallel jobs")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more info only for debugging purposes. You dont want to use it stdout - it will be cluttered.",
    )
    args = parser.parse_args()
    main(args)
