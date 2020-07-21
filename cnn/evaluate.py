import argparse
import torch
import copy

from dataset.dataset_utils import *
from save_utils import *


@torch.no_grad()
def evaluate(model, groups, device):
    transform_factory = ImageTransformFactory("test")
    model.eval()
    for group in groups:
        group = copy.deepcopy(group)
        metrics = []
        for pos in group.positions:
            microscope = Microscope(copy.deepcopy(group), pos.pos_idx, transform_factory.get_transform())
            x = microscope.current_image
            x = x.to(device)
            y = np.around(torch.mean(model(x)).cpu().numpy() * 50)
            microscope.move(y)
            metrics.append([
                np.abs(microscope.idx_distance_to_peak()),
                microscope.idx_distance_to_peak() == 0,
                np.abs(microscope.idx_distance_to_peak() <= 1),
                1
            ])
        # print(group.name, "l1 loss {:9.6f} accuracy {:9.6f} deep of field[-1 <= distance <= 1] accuracy {:9.6f} "
        #       "iterations {}".format(*np.mean(metrics, axis=0)))
        print("{} 1 {:9.6f} {:9.6f} {:9.6f} {}".format(group.name, *np.mean(metrics, axis=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model")
    args = parser.parse_args()

    from config import get_config
    config = get_config()
    config["network_type"] = "cnn_rgb"
    config["feature_type"] = "cnn_features"
    groups = []
    for json_file in config["test_dataset_json_files"]:
        groups.append(load_group_json(json_file, config["dataset_dir"]))

    device = torch.device("cuda", config["gpu_devices"] if config["gpu_devices"] else 0)
    device_cpu = torch.device("cpu")
    device = device if torch.cuda.is_available() else device_cpu

    log_dir = "{}-{}".format(config["network_type"], config["feature_type"])
    log_dir = os.path.join(config["log_dir"], log_dir)

    from cnn.train import MyModel
    model = MyModel()

    def load_fn(filepath):
        if os.path.exists(filepath):
            state_dicts = torch.load(filepath, map_location="cpu")
            model.load_state_dict(state_dicts["model_state_dict"])
        else:
            raise FileNotFoundError(filepath)
    cpkt = CheckPoint(log_dir, None, load_fn, clean=False)

    if args.best_model is not None:
        print(args.best_model)
        state_dicts = torch.load(args.best_model, map_location="cpu")
        if state_dicts.get("model_state_dict"):
            model.load_state_dict(state_dicts["model_state_dict"])
            print("load from checkpoint", args.best_model)
        else:
            model.load_state_dict(torch.load(args.best_model, map_location="cpu"))
            print("load from", args.best_model)
    else:
        cpkt.try_load_last()
    model.to(device)

    evaluate(model, groups, device)
