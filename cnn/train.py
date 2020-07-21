import sys
import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, resnet18
import torch.nn.functional as F

from dataset.dataset_utils import *
from save_utils import *


def train(model, criterion, optimizer, scheduler, config, checkpoint, logger):
    epoch = config["begin_epoch"]
    best_model = copy.deepcopy(model.state_dict())
    best_loss = sys.maxsize
    best_epoch = epoch - 1

    for epoch in range(epoch, config["epochs"]):
        print("Begin epoch")
        begin_time = time.time()

        print("-" * 50)
        print("Epoch {}/{}".format(epoch, config["epochs"] - 1))
        print("learning rate", scheduler.get_lr())

        model.train()
        train_losses = []
        iteration = 0
        for samples in train_data_loader:
            optimizer.zero_grad()

            images, distances = AutoFocusDataLoader.to_images_and_distances(samples, config["feature_type"])
            images = torch.stack(images, dim=0)
            images = images.to(device)
            distances = torch.from_numpy(np.array(distances, dtype=np.float32) * 0.02).to(device)
            y = model(images)
            loss = criterion(y, distances)
            _loss = loss.detach().cpu().numpy()
            _loss = float(_loss)
            train_losses.append(_loss * len(samples))
            loss.backward()
            optimizer.step()
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "Epoch {:03d} Iteration {:04d} loss {:6.4f}".format(epoch, iteration, _loss))
            iteration += 1

        losses = [np.sum(train_losses) / len(train_dataset), ]
        print("Average train loss {}".format(losses[0]))
        logger.add_scalar("train/loss", losses[0])
        save_file = "checkpoint-epoch{:02d}.pth".format(epoch)
        checkpoint.save(save_file, model, optimizer, scheduler, epoch)
        scheduler.step()

        model.eval()
        val_l1_losses = []
        for samples in val_data_loader:
            with torch.no_grad():
                images, distances = AutoFocusDataLoader.to_images_and_distances(samples, config["feature_type"])
                images = torch.stack(images, dim=0).to(device)
                distances = np.array(distances, dtype=np.float32) * 0.02
                y = model(images)
                l1_loss = np.abs(y.cpu().numpy() - distances)
                l1_loss = float(l1_loss)

                val_l1_losses.append(l1_loss * len(samples))
        losses = [np.sum(val_l1_losses) / len(val_dataset), ]
        print("Average val l1 loss {}".format(losses[0]))
        logger.add_scalar("val/l1_loss", losses[0])
        if best_loss > losses[0]:
            best_model = copy.deepcopy(model.state_dict())
            best_loss = losses[0]
            best_epoch = epoch

        end_time = time.time()
        print("Time of one epoch", (end_time - begin_time))

    return best_model, best_epoch


def collate_fn(batch):
    return batch


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        net = resnet50(pretrained=True)
        x_dim = net.fc.in_features
        layers = []
        for name, child in net.named_children():
            layers.append((name, child))
        model = nn.Sequential()
        for name, chlid in layers[:-1]:
            model.add_module(name, chlid)
        self.resnet = model
        self.fc = nn.Linear(x_dim, x_dim)
        self.reg = nn.Linear(x_dim, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        y = self.reg(x)
        return y


if __name__ == '__main__':
    from config import get_config

    config = get_config()
    config["network_type"] = "cnn_rgb"
    config["feature_type"] = "cnn_features"

    train_dataset = AutoFocusDataset(config["train_dataset_json_files"], config["dataset_dir"], mode="train")
    val_dataset = AutoFocusDataset(config["val_dataset_json_files"], config["dataset_dir"], mode="val")

    train_data_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                   num_workers=config["num_workers"],
                                   collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config["num_workers"],
                                 collate_fn=collate_fn)

    device = torch.device("cuda", config["gpu_devices"] if config["gpu_devices"] else 0)
    device_cpu = torch.device("cpu")
    device = device if torch.cuda.is_available() else device_cpu
    print(device)

    log_dir = "{}-{}".format(config["network_type"], config["feature_type"])
    log_dir = os.path.join(config["log_dir"], log_dir)
    logger = SummaryWriter(log_dir=log_dir)

    begin_epoch = 0
    model = MyModel()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=config["wd"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["lr_milestones"], gamma=0.1)


    def save_fn(filepath, model, optimizer, scheduler, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, filepath)
        print("Save checkpoint in {}".format(filepath))


    def load_fn(filepath):
        if os.path.exists(filepath):
            state_dicts = torch.load(filepath, map_location="cpu")
            global begin_epoch
            begin_epoch = state_dicts["epoch"] + 1
            model.load_state_dict(state_dicts["model_state_dict"])
            optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
            scheduler.load_state_dict(state_dicts["scheduler_state_dict"])
        else:
            raise FileNotFoundError(filepath)
        model.to(device)


    cpkt = CheckPoint(log_dir, save_fn, load_fn, clean=True)
    cpkt.try_load_last()
    for name, p in model.named_parameters():
        print(p.requires_grad, name)

    print("Begin")
    config["begin_epoch"] = begin_epoch
    best_model, best_epoch = train(model, criterion, optimizer, scheduler, config, cpkt, logger)
    logger.close()
    torch.save(best_model, os.path.join(log_dir, "best_model-epoch{}.pth".format(best_epoch)))
