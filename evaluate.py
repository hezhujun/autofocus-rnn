import argparse
import torch
import copy

from dataset.dataset_utils import *
from save_utils import *


def get_feature(microscopy, config):
    if config["feature_type"] == "cnn_features":
        images = microscopy.current_image
        return images
    elif config["feature_type"] == "focus_measures":
        p = microscopy.current_position
        feature = p.focus_measures_to_feature_vector(FOCUS_MEASURE_TYPES)
        feature = np.array(feature, dtype=np.float32)
        mu, sigma = calc_feature_mean_and_std()
        feature = (feature - mu) / sigma
        feature = torch.from_numpy(feature)
        features = feature.view(1, -1).to(device)
        return features
    else:
        assert False


@torch.no_grad()
def evaluate(model, groups, device, config):
    transform_factory = ImageTransformFactory("test")
    model.eval()
    for group in groups:
        group = copy.deepcopy(group)
        metrics = [[] for _ in range(config["rnn_len"])]
        for pos in group.positions:
            # print(group.name, pos)
            microscope = Microscope(copy.deepcopy(group), pos.pos_idx, transform_factory.get_transform())
            x = get_feature(microscope, config).to(device)
            h = torch.zeros(x.shape[0], config["a_dim"]).to(device)
            h, y = model(x, h, 0)
            y = np.around(torch.mean(y).cpu().numpy() * 50).astype(np.int)
            microscope.move(y)
            metrics[0].append([
                np.abs(microscope.idx_distance_to_peak()),
                microscope.idx_distance_to_peak() == 0,
                np.abs(microscope.idx_distance_to_peak()) <= 1,
                1
            ])
            for i in range(1, config["rnn_len"]):
                if y == 0:
                    metrics[i].append([
                        np.abs(microscope.idx_distance_to_peak()),
                        microscope.idx_distance_to_peak() == 0,
                        np.abs(microscope.idx_distance_to_peak()) <= 1,
                        metrics[i-1][-1][3],
                    ])
                    continue
                x = get_feature(microscope, config).to(device)
                h, y = model(x, h, i)
                y = np.around(torch.mean(y).cpu().numpy()*50).astype(np.int)
                microscope.move(y)
                metrics[i].append([
                        np.abs(microscope.idx_distance_to_peak()),
                        microscope.idx_distance_to_peak() == 0,
                        np.abs(microscope.idx_distance_to_peak()) <= 1,
                        i + 1,
                    ])
            # microscope.history()
        # print(group.name, "l1 loss {:9.6f} accuracy {:9.6f} deep of field[-1 <= distance <= 1] accuracy {:9.6f} "
        #       "iterations {}".format(*np.mean(metrics, axis=0)))
        print(group.name, end=" ")
        for i in range(0, config["rnn_len"]):
            print("{} {:9.6f} {:9.6f} {:9.6f} {}".format(i + 1, *np.mean(metrics[i], axis=0)), end=" ")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model")
    args = parser.parse_args()

    from config import get_config
    config = get_config()
    groups = []
    for json_file in config["test_dataset_json_files"]:
        groups.append(load_group_json(json_file, config["dataset_dir"]))

    device = torch.device("cuda", config["gpu_devices"] if config["gpu_devices"] else 0)
    device_cpu = torch.device("cpu")
    device = device if torch.cuda.is_available() else device_cpu

    log_dir = "{}-{}".format(config["network_type"], config["feature_type"])
    log_dir = os.path.join(config["log_dir"], log_dir)

    from network.model_rnn1 import MyModule
    model = MyModule(config["a_dim"], config["feature_type"], config["feature_len"])

    def load_fn(filepath):
        if os.path.exists(filepath):
            state_dicts = torch.load(filepath, map_location="cpu")
            model.load_state_dict(state_dicts["model_state_dict"])
            print("load from checkpoint", filepath)
        else:
            raise FileNotFoundError(filepath)
    cpkt = CheckPoint(log_dir, None, load_fn, clean=False)

    if args.best_model is not None:
        print("load from model", args.best_model)
        state_dicts = torch.load(args.best_model, map_location="cpu")
        if state_dicts.get("model_state_dict"):
            model.load_state_dict(state_dicts["model_state_dict"])
        else:
            model.load_state_dict(state_dicts)
    else:
        cpkt.try_load_last()
    model.to(device)

    evaluate(model, groups, device, config)
