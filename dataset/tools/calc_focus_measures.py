###############################################################################
# calculate the focus measures of a specify group
###############################################################################

import argparse
import os
from focus_measure_types import FOCUS_MEASURE_TYPES
from group_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("load_path", help="the path to load group object")
    parser.add_argument("save_path", help="the path to save group object")
    args = parser.parse_args()

    command = "/home/hzj/Documents/wsi_autofocus/dataset/focus_measures %s %d"

    group = load_group_json(args.load_path)

    print(group.name)
    for p in group.positions:
        print(p.pos_idx, ": ", end="")
        for i in range(1, len(FOCUS_MEASURE_TYPES)):
            print(i, end=" ")
            if p.focus_measures.get(FOCUS_MEASURE_TYPES[i]) is not None:
                continue
            image_path = os.path.join(p.dirname, p.filename)
            focus_measure = float(os.popen(command % (image_path, i)).read())
            p.focus_measures[FOCUS_MEASURE_TYPES[i]] = focus_measure
        print()

        dump_group_json(group, args.save_path)

