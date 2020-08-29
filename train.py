import argparse
import copy
import sys
import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import network.model_rnn1 as model_rnn
from dataset.dataset_utils import *
from save_utils import *
from torch.utils.tensorboard import SummaryWriter


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

            h = torch.zeros(len(samples), config["a_dim"]).to(device)
            loss = []
            for i in range(0, config["iter_len"]):
                images, distances = AutoFocusDataLoader.to_images_and_distances(samples, config["feature_type"])
                images = torch.stack(images, dim=0)
                images = images.to(device)
                distances = torch.from_numpy(np.array(distances, dtype=np.float32) * 0.02).to(device)
                h, y = model(images, h, i)
                loss.append(criterion(y, distances))
                _y = np.around(y.detach().cpu().numpy() * 50).astype(np.int64)
                AutoFocusDataLoader.move(samples, _y)

            loss = sum(loss)
            # loss = loss[-1]
            _loss = loss.detach().cpu().numpy()
            _loss = float(_loss)
            train_losses.append(_loss * len(samples))
            loss.backward()
            optimizer.step()
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:03d} Iteration {:04d} loss {:6.4f}".format(epoch, iteration, _loss))
            iteration += 1

        losses = [np.sum(train_losses) / len(train_dataset), ]
        print("Average train loss {}".format(losses[0]))
        logger.add_scalar("train/loss", losses[0])
        save_file = "checkpoint-epoch{:02d}.pth".format(epoch)
        checkpoint.save(save_file, model, optimizer, scheduler, epoch)
        scheduler.step()

        model.eval()
        val_l1_losses = []
        del distances
        del y
        for samples in val_data_loader:
            with torch.no_grad():
                h = torch.zeros(len(samples), config["a_dim"]).to(device)
                for i in range(0, config["iter_len"]):
                    images, distances = AutoFocusDataLoader.to_images_and_distances(samples, config["feature_type"])
                    images = torch.stack(images, dim=0).to(device)
                    distances = np.array(distances, dtype=np.float32)
                    h, y = model(images, h, i)
                    _y = np.around(y.detach().cpu().numpy() * 50).astype(np.int64)
                    if _y[0] == 0:
                        break
                    else:
                        AutoFocusDataLoader.move(samples, _y)

                l1_loss = np.abs(_y - distances)
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


# class BatchCollator(object):
#     def __call__(self, batch):
#         ms = []
#         for group, transformed_images, pos_idxs in batch:
#             for pos_idx in pos_idxs:
#                 m = Microscope(copy.deepcopy(group), pos_idx)
#                 for pos, image in zip(m.group.positions, transformed_images):
#                     pos.transformed_image = image
#                 ms.append(m)
#         return ms

class BatchCollator(object):
    def __call__(self, batch):
        return batch


if __name__ == '__main__':
    from config import get_config

    config = get_config()
    
    import json
    print(json.dumps(config, indent=2))

    train_dataset = AutoFocusDataset(config["train_dataset_json_files"], config["dataset_dir"], mode="train")
    val_dataset = AutoFocusDataset(config["val_dataset_json_files"], config["dataset_dir"], mode="val")

    train_data_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False,
                                   num_workers=config["num_workers"],
                                   collate_fn=BatchCollator())
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config["num_workers"],
                                 collate_fn=lambda batch: batch)

    device = torch.device("cuda", config["gpu_devices"] if config["gpu_devices"] else 0)
    device_cpu = torch.device("cpu")
    device = device if torch.cuda.is_available() else device_cpu
    print(device)

    log_dir = "{}-{}".format(config["network_type"], config["feature_type"])
    log_dir = os.path.join(config["log_dir"], log_dir)
    logger = SummaryWriter(log_dir=log_dir)

    begin_epoch = 0
    model = model_rnn.MyModule(config["a_dim"], config["feature_type"], config["feature_len"])

    print("load pretrain model from", config["pretrain_model"])
    state_dicts = torch.load(config["pretrain_model"], map_location="cpu")
    if state_dicts.get("model_state_dict"):
        model.load_state_dict(state_dicts["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state_dicts, strict=False)

    for p in model.cnn.parameters():
        p.requires_grad = False
    for i in range(config["iter_len"] - 1):
        for p in model.heads[i].parameters():
            p.requires_grad = False
    # if config["iter_len"] > 1 and model.feature_type == "cnn_features":
    #
    #     print("load pretrain model from", config["pretrain_model"])
    #     state_dicts = torch.load(config["pretrain_model"], map_location="cpu")
    #     if state_dicts.get("model_state_dict"):
    #         model.load_state_dict(state_dicts["model_state_dict"], strict=False)
    #     else:
    #         model.load_state_dict(state_dicts, strict=False)

    if config["iter_len"] == 1:
        criterion = nn.L1Loss()
    else:
        criterion = nn.L1Loss()
    model.to(device)

    # if config["iter_len"] == 1:
    #     optimizer = optim.Adam([
    #         {"params": model.cnn.parameters()},
    #         {"params": model.heads[0].parameters()},
    #         # {"params": model.models[0].parameters()},
    #     ], lr=config["learning_rate"], weight_decay=config["wd"])
    #     # optimizer = optim.SGD([
    #     #     {"params": model.cnn.parameters()},
    #     #     {"params": model.heads[0].parameters()},
    #     #     # {"params": model.models[0].parameters()},
    #     # ], lr=config["learning_rate"], momentum=0.9, weight_decay=config["wd"])
    #     for p in model.cnn.parameters():
    #         p.requires_grad = True
    #     for p in model.heads[0].parameters():
    #         p.requires_grad = True
    #     # for p in model.models[0].parameters():
    #     #     p.requires_grad = True
    # else:
    #     model.freeze_bn = True
    #     model.freeze_bn_affine = True
    #     last_iter = config["iter_len"] - 1
    #     optimizer = optim.Adam([
    #         {"params": model.heads[last_iter].parameters()},
    #         # {"params": model.models[last_iter].parameters()},
    #     ], lr=config["learning_rate"], weight_decay=config["wd"])
    #     # optimizer = optim.SGD([
    #     #     {"params": model.heads[last_iter].parameters()},
    #     #     # {"params": model.models[last_iter].parameters()},
    #     # ], lr=config["learning_rate"], momentum=0.9, weight_decay=config["wd"])
    #     for p in model.heads[last_iter].parameters():
    #         p.requires_grad = True
    #     # for p in model.models[last_iter].parameters():
    #     #     p.requires_grad = True
    # # if config["iter_len"] > 1 and model.feature_type == "cnn_features":
    # #     optimizer = optim.SGD([
    # #         {"params": model.heads[config["iter_len"] - 1].parameters()},
    # #     ], lr=config["learning_rate"], momentum=0.9, weight_decay=config["wd"])
    # # else:
    # #     optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=config["wd"])
    model.freeze_bn = True
    model.freeze_bn_affine = True
    train_module = model.heads[config["iter_len"] - 1]
    optimizer = optim.Adam(train_module.parameters(), lr=config["learning_rate"], weight_decay=config["wd"])
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
            model.to(device)
            print("load checkpoint from", filepath)
        else:
            raise FileNotFoundError(filepath)
    cpkt = CheckPoint(log_dir, save_fn, load_fn, clean=False)
    cpkt.try_load_last()
    for name, p in model.named_parameters():
        print(p.requires_grad, name)

    print("Begin")
    config["begin_epoch"] = begin_epoch
    best_model, best_epoch = train(model, criterion, optimizer, scheduler, config, cpkt, logger)
    logger.close()
    torch.save(best_model, os.path.join(log_dir, "best_model-epoch{}.pth".format(best_epoch)))
