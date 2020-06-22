import os

import torch


def search_checkpoint_records(save_dir):
    checkpoint_files = []
    if os.path.exists(save_dir):
        checkpoint_record = os.path.join(save_dir, "checkpoint.txt")
        if os.path.exists(checkpoint_record):
            for line in open(checkpoint_record):
                if line.strip() != "":
                    # print(line.strip())
                    checkpoint_files.append(line.strip())
    while len(checkpoint_files) > 5:
        try:
            print("remove {}".format(checkpoint_files[0]))
            os.remove(os.path.join(save_dir, checkpoint_files.pop(0)))
        except Exception:
            pass

    return checkpoint_files


def save_checkpoint_records(checkpoint_files, save_dir):
    with open(os.path.join(save_dir, "checkpoint.txt"), "w") as f:
        for checkpoint_file in checkpoint_files:
            f.write("{}\n".format(checkpoint_file))


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    save_dir = os.path.dirname(path)
    checkpoint_files = search_checkpoint_records(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(checkpoint_files) == 5:
        try:
            os.remove(os.path.join(save_dir, checkpoint_files.pop(0)))
        except Exception:
            pass
    checkpoint_files.append(os.path.basename(path))
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)

    save_checkpoint_records(checkpoint_files, save_dir)


def load_checkpoint(save_path):
    checkpoint_path = save_path
    if os.path.isdir(save_path) and os.path.exists(save_path):
        checkpoint_records = search_checkpoint_records(save_path)
        if len(checkpoint_records) == 0:
            raise FileNotFoundError(os.path.join(save_path, "checkpoint.txt"))
        checkpoint_path = os.path.join(save_path, checkpoint_records[-1])
    print("restore checkpoint from {}".format(checkpoint_path))
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location="cpu")
    else:
        raise FileNotFoundError(checkpoint_path)


def log_loss(losses, save_dir, epoch, mode="train"):
    assert isinstance(losses, list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "{}.txt".format(mode)), "a") as f:
        f.write("{} ".format(epoch))
        for loss in losses:
            f.write("{} ".format(loss))
        f.write("\n")
