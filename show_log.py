import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from config import get_config

config = get_config()

log_dir = "{}-{}".format(config["network_type"], config["feature_type"])
log_dir = os.path.join(config["log_dir"], log_dir)
parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=log_dir)
args = parser.parse_args()

if __name__ == '__main__':
    train_log_txt = os.path.join(args.dir, "train.txt")
    val_log_txt = os.path.join(args.dir, "val.txt")

    train_losses = np.loadtxt(train_log_txt)
    val_losses = np.loadtxt(val_log_txt)

    print(train_losses.shape)
    print(val_losses.shape)

    plt.plot(train_losses[:, 0], train_losses[:, 1], label="train")
    plt.plot(val_losses[:, 0], val_losses[:, 1], label="val")
    plt.title("l2 loss in {}".format(args.dir))
    plt.legend()

    plt.show()
