import torch
import numpy as np
from dataset.group_utils import *
from dataset.focus_measure_types import FOCUS_MEASURE_TYPES
from config import get_config
config = get_config()

mu = None
sigma = None


def calc_feature_mean_and_std():
    global mu, sigma
    if mu is not None and sigma is not None:
        # print("directly return mu and sigma")
        return mu, sigma

    groups = []
    for group_json in config["train_dataset_json_files"]:
        groups.append(load_group_json(group_json, config["dataset_dir"]))

    features = []
    for group in groups:
        for p in group.positions:
            feature = p.focus_measures_to_feature_vector(FOCUS_MEASURE_TYPES)
            assert len(feature) == config["feature_len"]
            features.append(feature)

    features = np.array(features, dtype=np.float32)
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)

    return mu, sigma
