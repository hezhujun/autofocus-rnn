import copy
import json

_default_config = {
    ###########################################################################
    # network architecture
    ###########################################################################
    "network_type": "CNN_ITERATOR",
    "feature_type": "cnn_features",  # "focus_measures", "cnn_features"
    "feature_len": 33,  # when feature_type=focus_measures

    ###########################################################################
    # dataset
    ###########################################################################
    "data_queue_len": 100,
    "dataset_dir": "/run/media/hezhujun/DATA1/Document/dataset/autofocus2",  # the directory of dataset
    # "dataset_dir": "/root/userfolder/datasets/autofocus2",  # the directory of dataset
    "train_dataset_json_files": [],
    "val_dataset_json_files": [],
    "test_dataset_json_files": [],

    ###########################################################################
    # LSTM parameters
    ###########################################################################
    "a_dim": 512,
    "iter_len": 5,

    ###########################################################################
    # GPUs config
    ###########################################################################
    "gpu_devices": 3,

    ###########################################################################
    # train hyper parameters
    ###########################################################################
    "image_split": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "learning_rate": 0.0001,
    "lr_milestones": [12, 16],
    "batch_size": 8,
    "epochs": 20,
    "wd": 1e-4,
    "num_workers": 8,
    "log_dir": "log/",
    # "pretrain_model": "CNN_ITERATOR3/checkpoint-epoch17.pth",
    # "pretrain_model": "CNN_ITERATOR5/checkpoint-epoch15.pth",
    # "pretrain_model": "CNN_ITERATOR6/checkpoint-epoch09.pth",
    # "pretrain_model": "CNN_ITERATOR18/checkpoint-epoch14.pth",
    "pretrain_model": "CNNGRU18/checkpoint-epoch00.pth",
}


_config = None


def load_json_list(json_file_list, json_files, pattern):
    with open(json_file_list, "r") as f:
        files = f.read()
        files = files.split()
        for file in files:
            file = file.strip()
            if file:
                json_files.append(pattern.format(file))


def get_config(**kwargs):
    global _config
    if _config is not None:
        return _config
    global _default_config
    config = copy.deepcopy(_default_config)

    for k, v in kwargs.items():
        config[k] = v

    pattern = "/run/media/hezhujun/DATA1/Document/dataset/autofocus2/info/{}.json"
    # pattern = "/root/userfolder/datasets/autofocus2/info/{}.json"
    load_json_list("dataset/train.txt", config["train_dataset_json_files"], pattern)
    load_json_list("dataset/val.txt", config["val_dataset_json_files"], pattern)
    load_json_list("dataset/test.txt", config["test_dataset_json_files"], pattern)

    # print(json.dumps(config, indent=2))
    _config = config
    return config
