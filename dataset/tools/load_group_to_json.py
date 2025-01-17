import os
import re
import argparse
from group_utils import *


def load_group(group_path_txt, group_name, pos_peak_idx, filename_pattern):
    with open(group_path_txt, "r") as f:
        image_paths = f.read().split("\n")
    group = Group(group_name)
    group.pos_peak_idx = pos_peak_idx
    for image_path in image_paths:
        group_path = os.path.dirname(image_path)
        group.path = group_path
        p = Position()
        p.dirname = group_path
        p.filename = os.path.basename(image_path)
        p.z_coordinate = int(re.match(filename_pattern, p.filename)[1])
        group.positions.append(p)

    if group.pos_peak_idx != -1:
        group.positions[group.pos_peak_idx].is_clear = True
    group.sort_positions()
    dump_group_json(group, group_name + ".json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group_path_txt", help="the path of a group txt file")
    parser.add_argument("group_name", help="the name of a group")
    args = parser.parse_args()

    group_path_txt = args.group_path_txt
    group_name = args.group_name

    load_group(group_path_txt, group_name, -1, r"(.*)\.jpg")
