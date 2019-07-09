import argparse

import network.model_rnn1 as model_rnn
from config import get_config
from dataset.dataset_utils import *
from save_utils import *

config = get_config()

parser = argparse.ArgumentParser()
parser.add_argument("--best_model")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_devices"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

model = model_rnn.MyMode(config["a_dim"], config["feature_type"], config["feature_len"])
model.to(device)

log_dir = "{}-{}".format(config["network_type"], config["feature_type"])
log_dir = os.path.join(config["log_dir"], log_dir)

transform = transforms.Compose([
    Normalize(),
    transforms.ToTensor(),
])


def get_feature(microscopy):
    if config["feature_type"] == "cnn_features":
        images = microscopy.current_image
        for i in range(len(images)):
            images[i] = transform(images[i])
        images = torch.stack(images, dim=0).to(device)
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


def evaluate(model, val_groups, T):
    model.eval()
    # metrics: focus error, focus accuracy, focus accuracy in depth of field, seaching times
    metrics = [[] for _ in range(T + 1)]  # 1 for the post stage
    for group in val_groups:
        for pos in group.positions:
            direction = 1
            microscope = Microscope(group, pos.pos_idx, FiveCrop(500))
            with torch.set_grad_enabled(False):
                x = get_feature(microscope)
                a = torch.zeros(x.size()[0], config["a_dim"]).to(device)
                c = torch.zeros(x.size()[0], config["a_dim"]).to(device)
                a, c, y = model(x, a, c, 0)
                y = np.around(np.mean(y.cpu().data.numpy()) * 50).astype(np.int)
                if y != 0:
                    direction = np.sign(y)
                microscope.move(y)
                metrics[0].append([
                    abs(microscope.idx_distance_to_peak()),
                    microscope.idx_distance_to_peak() == 0,
                    abs(microscope.idx_distance_to_peak() <= 1),
                    1,
                ])
                for iter in range(1, T):
                    if y == 0:
                        metrics[iter].append([
                            abs(microscope.idx_distance_to_peak()),
                            microscope.idx_distance_to_peak() == 0,
                            abs(microscope.idx_distance_to_peak()) <= 1,
                            metrics[iter - 1][-1][3],
                        ])
                        continue
                    x = get_feature(microscope)
                    a, c, y = model(x, a, c, iter)
                    y = np.around(np.mean(y.cpu().data.numpy()) * 50).astype(np.int)
                    if y != 0:
                        direction = np.sign(y)
                    microscope.move(y)
                    metrics[iter].append([
                        abs(microscope.idx_distance_to_peak()),
                        microscope.idx_distance_to_peak() == 0,
                        abs(microscope.idx_distance_to_peak()) <= 1,
                        iter + 1,
                    ])
            microscope.history()

            fm0 = microscope.get_current_focus_measure("SQUARED_GRADIENT")
            # print(microscope.current_position.pos_idx, fm0, end=" ")
            # survey the +1*direction position
            microscope.move(1 * direction)
            fm1 = microscope.get_current_focus_measure("SQUARED_GRADIENT")
            # print(microscope.current_position.pos_idx, fm1, end=" ")
            if fm0 >= fm1:
                # survey the -1*direction position
                microscope.move(-2 * direction)
                fm_1 = microscope.get_current_focus_measure("SQUARED_GRADIENT")
                # print(microscope.current_position.pos_idx, fm_1, end=" ")
                # We select the position with maximum fm value among fm_1, fm0 and fm1
                # but we needn't to make the final position located in the optimal position
                # because we only need the optimal image

                # We move the len in code for calculating metrics
                if fm0 > fm_1:
                    microscope.move(1 * direction)

                metrics[T].append([
                    abs(microscope.idx_distance_to_peak()),
                    microscope.idx_distance_to_peak() == 0,
                    abs(microscope.idx_distance_to_peak()) <= 1,
                    metrics[iter - 1][-1][3] + 2,
                ])
            else:
                # select the +1*direction position
                metrics[T].append([
                    abs(microscope.idx_distance_to_peak()),
                    microscope.idx_distance_to_peak() == 0,
                    abs(microscope.idx_distance_to_peak()) <= 1,
                    metrics[iter - 1][-1][3] + 1,
                ])
            # print()

    return np.array(metrics, dtype=np.float)


if __name__ == '__main__':
    groups = []
    for json_file in config["val_dataset_json_files"]:
        groups.append(load_group_json(json_file, config["dataset_dir"]))

    if args.best_model is not None:
        print(args.best_model)
        model.load_state_dict(torch.load(os.path.join(log_dir, args.best_model)))
    else:
        checkpoint = load_checkpoint(log_dir)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    T = 5
    metrics = evaluate(model, groups, T)
    for i in range(T):
        print(
            "T={} "
            "l1 loss {:9.6f} "
            "accuracy {:9.6f} "
            "deep of field[-1 <= distance <= 1] accuracy {:9.6f} "
            "searching times {}".format(i + 1, *np.mean(metrics[i], axis=0)))

    print(
        "T={} "
        "l1 loss {:9.6f} "
        "accuracy {:9.6f} "
        "deep of field[-1 <= distance <= 1] accuracy {:9.6f} "
        "searching times {}".format(T + 1, *np.mean(metrics[T], axis=0)))

    for i in range(T):
        print("{}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{}".format(i + 1, *np.mean(metrics[i], axis=0)))

    print("{}\t{:9.6f}\t{:9.6f}\t{:9.6f}\t{}".format(T + 1, *np.mean(metrics[T], axis=0)))
