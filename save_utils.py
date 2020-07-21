import os

import torch


class CheckPoint(object):

    def __init__(self, root, save_fn, load_fn, record_filename="checkpoint.txt", num_record=-1, clean=False):
        self.root = root
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.filename = record_filename
        self.num_record = num_record

        if not os.path.exists(root):
            os.makedirs(root)

        # clean the record file
        record_filename = os.path.join(self.root, self.filename)
        if clean and os.path.exists(record_filename):
            with open(record_filename, "w") as f:
                f.write('')

    def read_record_file(self):
        record_filename = os.path.join(self.root, self.filename)
        if not os.path.exists(record_filename):
            return []
        with open(record_filename, "r") as f:
            records = f.read().split()
            records = [i for i in records if i.strip()]
            return records

    def write_record_file(self, records):
        record_filename = os.path.join(self.root, self.filename)
        with open(record_filename, "w") as f:
            for record in records:
                f.write("{}\n".format(record))

    def save(self, filename, *args):
        if not os.path.isabs(filename):
            filename = os.path.join(self.root, filename)
        self.save_fn(filename, *args)
        records = self.read_record_file()
        records.append(filename)

        if 0 < self.num_record < len(records):
            while self.num_record < len(records):
                record = records.pop(0)
                try:
                    os.remove(record)
                except Exception:
                    pass

        self.write_record_file(records)

    def load(self, filename):
        self.load_fn(filename)

    def try_load_last(self):
        record_filename = os.path.join(self.root, self.filename)
        if not os.path.exists(record_filename):
            return
        with open(record_filename, "r") as f:
            records = f.read().split()
            records = [i for i in records if i.strip()]

        if len(records) > 0:
            self.load_fn(records[-1])


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
