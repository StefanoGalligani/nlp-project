#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocesses predictions json and makes sure that it is in the correct format for evaluation.
Removes non-matching keys, or maps to closest ones specific to the dataset.
"""

import argparse
import json
import sys
import Levenshtein
from copy import deepcopy

# Schemas for MultiWOZ and SpokenWoz

SA_MWOZ = {
    "hotel": [
        "area",
        "day",
        "internet",
        "name",
        "parking",
        "people",
        "pricerange",
        "stars",
        "stay",
        "type",
    ],
    "train": ["arriveby", "day", "departure", "destination", "leaveat", "people"],
    "attraction": ["area", "name", "type"],
    "restaurant": ["area", "day", "food", "name", "people", "pricerange", "time"],
    "taxi": ["arriveby", "departure", "destination", "leaveat"],
    "hospital": ["department"],
}

SP_WOZ = {
    "restaurant": ["area", "day", "food", "name", "people", "pricerange", "time"],
    "profile": ["email", "idnumber", "name", "phonenumber", "platenumber"],
    "hotel": [
        "area",
        "day",
        "internet",
        "name",
        "parking",
        "people",
        "pricerange",
        "stars",
        "stay",
        "type",
    ],
    "taxi": ["arriveby", "departure", "destination", "leaveat"],
    "train": ["arriveby", "day", "departure", "destination", "leaveat", "people"],
    "attraction": ["area", "name", "type"],
    "hospital": ["department"],
}


def preprocess_schema(data, schema, verbose=False):
    """Preprocesses the schema to remove any slot that isn't part of the schema"""

    new_data = {}
    flag = True

    for dialog_id, turn_list in data.items():
        if verbose:
            print("\rProcessing", dialog_id, end=" ")
        new_data[dialog_id] = []  # deepcopy(turn_list)

        for i, turn_dict in enumerate(turn_list):
            new_turn_dict = deepcopy(turn_dict)
            new_data[dialog_id].append(new_turn_dict)

            for act_dom in turn_dict["active_domains"]:
                if act_dom not in schema:
                    # do not keep any active domain that isn't part of the  schema
                    if verbose:
                        print(f"\n{i} {act_dom} not found in active_domain schema. Removing.")
                    new_data[dialog_id][i]["active_domains"].remove(act_dom)

            if len(turn_dict["state"]) == 0:
                continue

            for domain, domain_dict in turn_dict["state"].items():
                if domain not in schema:
                    if verbose:
                        print(f"{i} {domain} not found in schema. Removing.")
                    new_data[dialog_id][i]["state"].pop(domain)
                    continue

                if isinstance(domain_dict, str):
                    if verbose:
                        print(f"{i} {domain}:: {domain_dict} is string. Removing.")
                    new_data[dialog_id][i]["state"].pop(domain)
                    continue

                for slot_key, slot_val in domain_dict.items():
                    if slot_key not in schema[domain]:
                        flag = False
                        if verbose:
                            print(f"{i} {domain}:: {slot_key}:{slot_val}")

                        # Find closest slot key
                        for gt_slot_key in schema[domain]:
                            if Levenshtein.seqratio(slot_key, gt_slot_key) > 0.7:
                                if verbose:
                                    print(f"  {slot_key} --> {gt_slot_key}")

                                new_data[dialog_id][i]["state"][domain][gt_slot_key] = new_data[dialog_id][i][
                                    "state"
                                ][domain].pop(slot_key)
                                flag = True
                                break

                        if not flag:
                            if verbose:
                                print(f"  Deleting {slot_key}")
                            del new_data[dialog_id][i]["state"][domain][slot_key]

    return new_data


def main(args):
    """main method"""

    if len(sys.argv) > 2:
        data = {}
        with open(args.in_json, "r", encoding="utf-8") as f:
            data = json.load(f)

    else:
        data = json.load(sys.stdin)

    SCHEMA = SA_MWOZ if args.sa else SP_WOZ

    new_data = preprocess_schema(data, SCHEMA, args.verbose)

    # flag = True

    # for dialog_id, turn_list in data.items():
    #     if args.verbose:
    #         print("\rProcessing", dialog_id, end=" ")
    #     new_data[dialog_id] = []  # deepcopy(turn_list)

    #     for i, turn_dict in enumerate(turn_list):
    #         new_turn_dict = deepcopy(turn_dict)
    #         new_data[dialog_id].append(new_turn_dict)

    #         for act_dom in turn_dict["active_domains"]:
    #             if act_dom not in SCHEMA:
    #                 # do not keep any active domain that isn't part of the  schema
    #                 if args.verbose:
    #                     print(f"\n{i} {act_dom} not found in active_domain schema. Removing.")
    #                 new_data[dialog_id][i]["active_domains"].remove(act_dom)

    #         if len(turn_dict["state"]) == 0:
    #             continue

    #         for domain, domain_dict in turn_dict["state"].items():
    #             if domain not in SCHEMA:
    #                 if args.verbose:
    #                     print(f"{i} {domain} not found in schema. Removing.")
    #                 new_data[dialog_id][i]["state"].pop(domain)
    #                 continue

    #             for slot_key, slot_val in list(domain_dict.items()):
    #                 if slot_key not in SCHEMA[domain]:
    #                     flag = False
    #                     if args.verbose:
    #                         print(f"{i} {domain}:: {slot_key}:{slot_val}")

    #                     # Find closest slot key
    #                     for gt_slot_key in SCHEMA[domain]:
    #                         if Levenshtein.seqratio(slot_key, gt_slot_key) > 0.7:
    #                             if args.verbose:
    #                                 print(f"  {slot_key} --> {gt_slot_key}")

    #                             new_data[dialog_id][i]["state"][domain][gt_slot_key] = new_data[dialog_id][i][
    #                                 "state"
    #                             ][domain].pop(slot_key)
    #                             flag = True
    #                             break

    #                     if not flag:
    #                         if args.verbose:
    #                             print(f"  Deleting {slot_key}")
    #                         del new_data[dialog_id][i]["state"][domain][slot_key]

    #     print(line)
    #     ctr += 1
    #     break
    if args.verbose:
        print()

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4)
    else:
        json.dump(new_data, sys.stdout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    me_group = parser.add_mutually_exclusive_group(required=True)
    me_group.add_argument("--sa", action="store_true", help="Bool flag required for MultiWOZ dataset")
    me_group.add_argument("--sp", action="store_true", help="Bool flag required for SpokenWoz dataset")
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
        "--verbose",
        action="store_true",
        help="Print more info only for debugging purposes. You dont want to use it stdout - it will be cluttered.",
    )
    args = parser.parse_args()
    main(args)
