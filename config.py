import copy

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
    "data_queue_len": 500,
    "dataset_dir": "/run/media/hezhujun/DATA1/Document/dataset/autofocus",  # the directory of dataset
    "dataset_json_files": [
        "dataset/data_json/group01.json",
        "dataset/data_json/group02.json",
        "dataset/data_json/group03.json",
        "dataset/data_json/group04.json",
        "dataset/data_json/group05.json",
        "dataset/data_json/group06.json",
        "dataset/data_json/group07.json",
        "dataset/data_json/group08.json",
        "dataset/data_json/group09.json",
        "dataset/data_json/group10.json",
        "dataset/data_json/group11.json",
        "dataset/data_json/group12.json",
        "dataset/data_json/group13.json",
        "dataset/data_json/group14.json",
        "dataset/data_json/group15.json",
        "dataset/data_json/group16.json",
        "dataset/data_json/group17.json",
        "dataset/data_json/group18.json",
        "dataset/data_json/group19.json",
        "dataset/data_json/group20.json",
    ],
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


def get_config(**kwargs):
    global _config
    if _config is not None:
        return _config
    global _default_config
    config = copy.deepcopy(_default_config)

    for k, v in kwargs.items():
        config[k] = v

    for k, v in config.items():
        if k == "dataset_json_files":
            continue
        print(k, v)

    config["train_dataset_json_files"] = []
    config["val_dataset_json_files"] = []

    for i in range(len(config["dataset_json_files"])):
        if i == (config["val_group"] - 1):
            config["val_dataset_json_files"].append(config["dataset_json_files"][i])
        else:
            config["train_dataset_json_files"].append(config["dataset_json_files"][i])
    config["log_dir"] = "log/valgroup%02d" % (config["val_group"])
    _config = config
    return config
