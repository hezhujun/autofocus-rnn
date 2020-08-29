import os
import pickle
from torchvision import transforms
from dataset.group_utils import *
from PIL import Image
from config import get_config


def split(group, ratios, transform):
    for x_ratio in ratios:
        for y_ratio in ratios:
            x1, y1 = None, None
            sub_images = []
            for pos in group.positions:
                if pos.image is None:
                    image_path = os.path.join(pos.dirname, pos.filename)
                    image = Image.open(image_path)
                else:
                    image = pos.image
                W, H = image.size
                W = W - 500
                H = H - 500
                x1 = int(W * x_ratio)
                y1 = int(H * y_ratio)
                x2 = x1 + 500
                y2 = y1 + 500
                sub_image = image.crop((x1, y1, x2, y2))

                save_path = os.path.join(pos.dirname, "{}_{}_{}_{}_{}.png".format(pos.filename, x1, y1, 500, 500))
                print(save_path)
                sub_image.save(save_path)
            #     sub_image = transform(sub_image)
            #     sub_images.append(sub_image)
            #
            # save_path = os.path.join(group.abspath, "all_pos_{}_{}_{}_{}.pickle".format(x1, y1, 500, 500))
            # print(save_path)
            # with open(save_path, "wb") as f:
            #     pickle.dump(sub_images, f)


if __name__ == "__main__":
    config = get_config()
    group_json_files = config["test_dataset_json_files"]
    groups = []
    for group_json_file in group_json_files:
        groups.append(load_group_json(group_json_file, config["dataset_dir"]))

    ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    print("ratios", ratios)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0))
    ])
    for group in groups:
        print(group.name)
        split(group, ratios, transform)

