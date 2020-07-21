import json
import os

import skimage.io as io
from .load_data_queue import get_image


class Position(object):
    def __init__(self):
        self.dirname = ""  # don't save in json
        self.filename = ""
        self.z_coordinate = -1
        self.pos_idx = -1
        self.is_clear = False
        self.focus_measures = {}
        self.image = None

    def __repr__(self):
        return "Position {} index {}".format(self.z_coordinate, self.pos_idx)

    def get_image(self, transform):
        if self.image is None:
            self.image = get_image(os.path.join(self.dirname, self.filename))
            if transform is not None:
                self.image = transform(self.image)
        # img = io.imread(os.path.join(self.dirname, self.filename))
        return self.image

    def focus_measures_to_feature_vector(self, types):
        vector = []
        for type in types:
            vector.append(self.focus_measures[type])
        return vector


class Group(object):
    def __init__(self, name):
        self.path = ""
        self.name = name
        self.pos_number = -1
        self.pos_peak_idx = -1
        self.positions = []  # sorted

    @property
    def abspath(self):
        if os.path.isabs(self.path):
            return self.path
        else:
            return os.path.abspath(self.path)

    def sort_positions(self):
        self.positions.sort(key=lambda p: p.z_coordinate)
        for i in range(len(self.positions)):
            self.positions[i].pos_idx = i
        self.pos_number = len(self.positions)

    def __repr__(self):
        return "Group {} {} position(s)".format(self.name, self.pos_number)


def dump_group_json(group, save_path):
    group_dict = {}
    group_dict["path"] = group.path
    group_dict["name"] = group.name
    group_dict["pos_number"] = group.pos_number
    group_dict["pos_peak_idx"] = group.pos_peak_idx
    group_dict["positions"] = []

    for p in group.positions:
        pos_dict = {}
        pos_dict["filename"] = p.filename
        pos_dict["z_coordinate"] = p.z_coordinate
        pos_dict["pos_idx"] = p.pos_idx
        pos_dict["is_clear"] = p.is_clear
        pos_dict["focus_measures"] = p.focus_measures
        group_dict["positions"].append(pos_dict)
    with open(save_path, "w") as f:
        json.dump(group_dict, f, indent=2, sort_keys=True)


def load_group_json(load_path, root=None):
    with open(load_path, "r") as f:
        group_dict = json.load(f)

    group = Group(group_dict["name"])
    if root is not None:
        group.path = os.path.join(root, group_dict["name"])
    else:
        group.path = group_dict["path"]
    group.pos_number = group_dict["pos_number"]
    group.pos_peak_idx = group_dict["pos_peak_idx"]
    for pos_dict in group_dict["positions"]:
        p = Position()
        p.filename = pos_dict["filename"]
        p.z_coordinate = pos_dict["z_coordinate"]
        p.pos_idx = pos_dict["pos_idx"]
        p.is_clear = pos_dict["is_clear"]
        p.focus_measures = pos_dict["focus_measures"]
        p.dirname = group.abspath
        group.positions.append(p)
    group.positions[group.pos_peak_idx].is_clear = True
    return group
