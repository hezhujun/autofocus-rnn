import copy
import json

_default_config = {
    ###########################################################################
    # network architecture
    ###########################################################################
    "network_type": "RNN-1",
    "feature_type": "cnn_features",  # "focus_measures", "cnn_features"
    "feature_len": 33,  # when feature_type=focus_measures

    ###########################################################################
    # dataset
    ###########################################################################
    "data_queue_len": 100,
    "dataset_dir": "/run/media/hezhujun/DATA1/Document/dataset/autofocus",  # the directory of dataset
    "train_dataset_json_files": [],
    "val_dataset_json_files": [],
    "test_dataset_json_files": [],
    "val_group": 1,

    ###########################################################################
    # LSTM parameters
    ###########################################################################
    "a_dim": 100,
    "rnn_len": 5,

    ###########################################################################
    # GPUs config
    ###########################################################################
    "gpu_devices": "0",

    ###########################################################################
    # train hyper parameters
    ###########################################################################
    "learning_rate": 0.001,
    "lr_milestones": [100, 150],
    "batch_size": 5,
    "epochs": 200,
    "wd": 0.0001,
    "num_workers": 4,

}


_config = None

def load_json_list(json_file_list, json_files):
    with open(json_file_list, "r") as f:
        files = f.read()
        files = files.split()
        for file in files:
            file = file.strip()
            if file:
                json_files.append("dataset/data_json/{}.json".format(file))


def get_config(**kwargs):
    global _config
    if _config is not None:
        return _config
    global _default_config
    config = copy.deepcopy(_default_config)

    for k, v in kwargs.items():
        config[k] = v

    load_json_list("dataset/train.txt", config["train_dataset_json_files"])
    load_json_list("dataset/val.txt", config["val_dataset_json_files"])
    load_json_list("dataset/test.txt", config["test_dataset_json_files"])

    print(json.dumps(config, indent=2))

    config["log_dir"] = "log/"
    _config = config
    return config
