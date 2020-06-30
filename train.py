import argparse
import copy
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--network_type")
parser.add_argument("--feature_type")
parser.add_argument("--feature_len")
parser.add_argument("--data_queue_len", type=int)
parser.add_argument("--dataset_dir")
parser.add_argument("--val_group", type=int)
parser.add_argument("--a_dim", type=int)
parser.add_argument("--gpu_devices")
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--wd", type=float)
parser.add_argument("--num_workers", type=int)
args = parser.parse_args()

cfg = {}
if args.network_type:
    cfg["network_type"] = args.network_type
if args.feature_type:
    cfg["feature_type"] = args.feature_type
if args.feature_len:
    cfg["feature_len"] = args.feature_len
if args.data_queue_len:
    cfg["data_queue_len"] = args.data_queue_len
if args.dataset_dir:
    cfg["dataset_dir"] = args.dataset_dir
if args.val_group:
    cfg["val_group"] = args.val_group
if args.a_dim:
    cfg["a_dim"] = args.a_dim
if args.gpu_devices:
    cfg["gpu_devices"] = args.gpu_devices
if args.learning_rate:
    cfg["learning_rate"] = args.learning_rate
if args.batch_size:
    cfg["batch_size"] = args.batch_size
if args.epochs:
    cfg["epochs"] = args.epochs
if args.wd:
    cfg["wd"] = args.wd
if args.num_workers:
    cfg["num_workers"] = args.num_workers

from config import get_config

config = get_config(**cfg)

import torch.nn as nn
import torch.optim as optim

import network.model_rnn1 as model_rnn
from dataset.dataset_utils import *
from save_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_devices"]

train_dataset = AutoFocusDataset(config["train_dataset_json_files"], config["dataset_dir"], mode="train")
val_dataset = AutoFocusDataset(config["val_dataset_json_files"], config["dataset_dir"], mode="val")
# test_dataset = AutoFocusDataset(config["test_dataset_json_files"], config["dataset_dir"], mode="test")

train_data_loader = AutoFocusDataLoader(train_dataset, config["batch_size"], config["feature_type"], True,
                                        ImageTransformFactory("train"))
val_data_loader = AutoFocusDataLoader(val_dataset, config["batch_size"], config["feature_type"], False,
                                      ImageTransformFactory("val"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

log_dir = "{}-{}".format(config["network_type"], config["feature_type"])
log_dir = os.path.join(config["log_dir"], log_dir)

model = model_rnn.MyMode(config["a_dim"], config["feature_type"], config["feature_len"])
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=config["wd"])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["lr_milestones"], gamma=0.1)


def train(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    epoch = 0

    checkpoint_file = os.path.join(log_dir, "checkpoint.txt")
    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(os.path.dirname(checkpoint_file))
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"] + 1

    best_model = copy.deepcopy(model.state_dict())
    best_loss = sys.maxsize
    best_epoch = epoch - 1

    for epoch in range(epoch, num_epochs):
        begin_time = time.time()

        print("-" * 50)
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("learning rate", scheduler.get_lr())

        model.train()
        train_losses = []
        for samples in train_data_loader:
            optimizer.zero_grad()

            one_losses = []
            images, distances = train_data_loader.to_images_and_distances(samples)
            images = torch.stack(images, dim=0).to(device)
            distances = torch.from_numpy(np.array(distances, dtype=np.float32) / 50).to(device)
            a = torch.zeros(len(samples), config["a_dim"]).to(device)
            c = torch.zeros(len(samples), config["a_dim"]).to(device)
            a, c, y = model(images, a, c, 0)
            one_losses.append(criterion(y, distances))

            for i in range(1, config["rnn_len"]):
                # update samples
                y = np.around(y.cpu().data.numpy() * 50).astype(np.int)
                train_data_loader.move(samples, y)
                images, distances = train_data_loader.to_images_and_distances(samples)
                images = torch.stack(images, dim=0).to(device)
                distances = torch.from_numpy(np.array(distances, dtype=np.float32) / 50).to(device)
                a, c, y = model(images, a, c, i)
                one_losses.append(criterion(y, distances))

            loss = sum(one_losses)
            _loss = float(loss.cpu().data.numpy()[0])
            train_losses.append(_loss * len(samples))
            # print("Train loss {}".format(_loss))
            loss.backward()
            optimizer.step()

        losses = [np.sum(train_losses) / len(train_dataset), ]
        print("Average train loss {}".format(losses[0]))
        log_loss(losses, log_dir, epoch, mode="train")
        save_path = os.path.join(log_dir, "checkpoint-epoch{:02d}.pth".format(epoch))
        save_checkpoint(model, optimizer, scheduler, epoch, save_path)
        print("Save checkpoint in {}".format(save_path))
        scheduler.step()

        model.eval()
        val_l1_losses = []
        for samples in val_data_loader:
            with torch.no_grad():
                images, distances = val_data_loader.to_images_and_distances(samples)
                images = torch.stack(images, dim=0).to(device)
                distances = torch.from_numpy(np.array(distances, dtype=np.float32) / 50).to(device)
                a = torch.zeros(len(samples), config["a_dim"]).to(device)
                c = torch.zeros(len(samples), config["a_dim"]).to(device)
                a, c, y = model(images, a, c, 0)

                for i in range(1, config["rnn_len"]):
                    y = np.around(y.cpu().data.numpy() * 50).astype(np.int)
                    val_data_loader.move(samples, y)
                    images, distances = val_data_loader.to_images_and_distances(samples)
                    images = torch.stack(images, dim=0).to(device)
                    distances = torch.from_numpy(np.array(distances, dtype=np.float32) / 50).to(device)
                    a, c, y = model(images, a, c, i)

                l1_loss = torch.mean(torch.abs(y - distances)).cpu().data.numpy()[0]
                # print("Val l1 loss {}".format(l1_loss))

                val_l1_losses.append(l1_loss * len(samples))
        losses = [np.sum(val_l1_losses) / len(val_dataset), ]
        print("Average val l1 loss {}".format(losses[0]))
        log_loss(losses, log_dir, epoch, mode="val")
        if best_loss > losses[0]:
            best_model = copy.deepcopy(model.state_dict())
            best_loss = losses[0]
            best_epoch = epoch

        end_time = time.time()
        print("Time of one epoch", (end_time - begin_time))

    return best_model, best_epoch


if __name__ == '__main__':
    best_model, best_epoch = train(model, criterion, optimizer, scheduler, num_epochs=config["epochs"])
    torch.save(best_model, os.path.join(log_dir, "best_model-epoch{}.pth".format(best_epoch)))
