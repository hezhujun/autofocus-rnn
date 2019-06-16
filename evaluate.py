import argparse
import network.model_rnn1 as model_rnn
from config import config
from dataset.dataset_utils import *
from save_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--best_model")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_devices"]

train_dataset = AutoFocusDataset(config["train_dataset_json_files"], config["dataset_dir"], mode="train")
val_dataset = AutoFocusDataset(config["val_dataset_json_files"], config["dataset_dir"], mode="val")

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


def evaluate(model, dataset, iteration):
    model.eval()
    metrics = [[] for i in range(iteration)]
    microscopies = []
    for group, pos_idx in dataset:
        microscopy = Microscope(group, pos_idx, FiveCrop(500))
        with torch.set_grad_enabled(False):
            x = get_feature(microscopy)
            a = torch.zeros(x.size()[0], config["a_dim"]).to(device)
            c = torch.zeros(x.size()[0], config["a_dim"]).to(device)
            a, c, y = model(x, a, c, 0)
            y = np.around(np.mean(y.cpu().data.numpy()) * 50).astype(np.int)
            microscopy.move(y)
            metrics[0].append([abs(microscopy.idx_distance_to_peak()), microscopy.idx_distance_to_peak() == 0, abs(microscopy.idx_distance_to_peak()) <= 1])
            for iter in range(1, iteration):
                x = get_feature(microscopy)
                a, c, y = model(x, a, c, iter)
                y = np.around(np.mean(y.cpu().data.numpy()) * 50).astype(np.int)
                microscopy.move(y)
                metrics[iter].append([abs(microscopy.idx_distance_to_peak()), microscopy.idx_distance_to_peak() == 0, abs(microscopy.idx_distance_to_peak()) <= 1])
            microscopy.history()
            microscopies.append(microscopy)

    return np.array(metrics)


if __name__ == '__main__':
    # microscopies = evaluate(model, val_dataset, 5, os.path.join(log_dir, "checkpoint.txt"))
    # l1_losses = []
    # accuracies = []
    # dof_acc = []
    # for m in microscopies:
    #     idx_distance = m.idx_distance_to_peak()
    #     l1_losses.append(abs(idx_distance))
    #     accuracies.append(idx_distance == 0)
    #     dof_acc.append(abs(idx_distance) <= 1)
    # print("l1 loss", np.mean(l1_losses))
    # print("accuray", np.mean(accuracies))
    # print("deep of field[-1 <= distance <= 1] accuracy", np.mean(dof_acc))

    if args.best_model is not None:
        print(args.best_model)
        model.load_state_dict(torch.load(os.path.join(log_dir, args.best_model)))
    else:
        checkpoint = load_checkpoint(log_dir)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    metrics = evaluate(model, val_dataset, 5)
    for i in range(5):
        print("iter {} l1 loss {:9.6f} accuracy {:9.6f} deep of field[-1 <= distance <= 1] accuracy {:9.6f}".format(i + 1, *np.mean(metrics[i], axis=0)))

    for i in range(5):
        print("{}\t{:9.6f}\t{:9.6f}\t{:9.6f}".format(i + 1, *np.mean(metrics[i], axis=0)))
