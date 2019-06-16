import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from focus_measures_norm import calc_feature_mean_and_std
from .focus_measure_types import FOCUS_MEASURE_TYPES
from .group_utils import *
from .microscope_utils import *


class AutoFocusDataset(Dataset):

    def __init__(self, group_json_paths, dataset_dir, mode="train"):
        self.groups = []
        for group_json_path in group_json_paths:
            self.groups.append(load_group_json(group_json_path, dataset_dir))

        self.mode = mode

        self.group_positions_lens = []
        for group in self.groups:
            self.group_positions_lens.append(len(group.positions))

        self.idx2group_position = {}
        i = 0
        for group in self.groups:
            for pos_idx in range(len(group.positions)):
                self.idx2group_position[i] = (group, pos_idx)
                i += 1

    def __len__(self):
        return int(np.sum(self.group_positions_lens))

    def __getitem__(self, idx):
        # assert idx < self.__len__()
        if idx >= self.__len__():
            raise IndexError
        group, pos_idx = self.idx2group_position[idx]
        return group, pos_idx


class AutoFocusDataLoader(object):

    def __init__(self, dataset, batch_size, feature_type, shuffle=True, transform_factory=None):
        self.dataset = dataset
        self.dataset_idxs = np.arange(len(dataset))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transform_factory = transform_factory
        self.feature_type = feature_type

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, idx):
        if idx == 0 and self.shuffle:
            np.random.shuffle(self.dataset_idxs)
        if idx >= self.__len__():
            raise IndexError
        begin = idx * self.batch_size
        end = begin + self.batch_size
        end = end if end <= len(self.dataset) else len(self.dataset)

        samples = []
        for idx in self.dataset_idxs[begin:end]:
            group, pos_idx = self.dataset[idx]
            transform = None
            if self.transform_factory:
                transform = self.transform_factory.get_transform()
            m = Microscope(group, pos_idx, transform)
            samples.append(m)
        return samples

    def to_images_and_distances(self, microscopes):
        samples = []
        distances = []
        for m in microscopes:
            sample = None
            if self.feature_type == "cnn_features":
                sample = (m.current_image)
            elif self.feature_type == "focus_measures":
                p = m.current_position
                feature = p.focus_measures_to_feature_vector(FOCUS_MEASURE_TYPES)
                feature = np.array(feature, dtype=np.float32)
                mu, sigma = calc_feature_mean_and_std()
                feature = (feature - mu) / sigma
                sample = (torch.from_numpy(feature))
            assert sample is not None
            samples.append(sample)
            distances.append(m.idx_distance_to_peak())
        return samples, distances

    def move(self, microscopes, distances):
        for m, distance in zip(microscopes, distances):
            # m.move(m.convert_distance_to_idx_distance(distance))
            m.move(distance)


class RandomCrop(object):

    def __init__(self, size, seed=None):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        random_state = np.random.get_state()
        if seed:
            np.random.seed(seed)
        self.x, self.y = np.random.rand(2)
        np.random.set_state(random_state)

    def __call__(self, image):
        shape = image.shape
        if len(shape) == 3:
            H, W, C = shape
            h = int(self.y * (H - self.size[0]))
            w = int(self.x * (W - self.size[1]))
            image = image[h:h + self.size[0], w:w + self.size[1], :]
        else:
            assert len(shape) == 2
            H, W = shape
            h = int(self.y * (H - self.size[0]))
            w = int(self.x * (W - self.size[1]))
            image = image[h:h + self.size[0], w:w + self.size[1]]
            image = image.reshape((self.size[0], self.size[1], 1))

        return image


class CenterCrop(object):

    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image):
        shape = image.shape
        if len(shape) == 3:
            H, W, C = shape
            h = int(0.5 * (H - self.size[0]))
            w = int(0.5 * (W - self.size[1]))
            image = image[h:h + self.size[0], w:w + self.size[1], :]
        else:
            assert len(shape) == 2
            H, W = shape
            h = int(0.5 * (H - self.size[0]))
            w = int(0.5 * (W - self.size[1]))
            image = image[h:h + self.size[0], w:w + self.size[1]]
            image = image.reshape((self.size[0], self.size[1], 1))

        return image


class FiveCrop(object):

    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.point_center = (0.5, 0.5)
        self.point1 = (0.3, 0.3)
        self.point2 = (0.3, 0.7)
        self.point3 = (0.7, 0.3)
        self.point4 = (0.7, 0.7)
        self.points = [self.point_center, self.point1, self.point2, self.point3, self.point4]

    def __call__(self, image):
        shape = image.shape
        images = []
        if len(shape) == 3:
            H, W, C = shape
            for point in self.points:
                h = int(point[0] * (H - self.size[0]))
                w = int(point[1] * (W - self.size[1]))
                sub_image = image[h:h + self.size[0], w:w + self.size[1], :]
                images.append(sub_image)
        else:
            assert len(shape) == 2
            H, W = shape
            for point in self.points:
                h = int(point[0] * (H - self.size[0]))
                w = int(point[1] * (W - self.size[1]))
                sub_image = image[h:h + self.size[0], w:w + self.size[1]]
                sub_image = sub_image.reshape((self.size[0], self.size[1], 1))
                images.append(sub_image)

        return images


class ImageTransformFactory(object):

    def __init__(self, mode="train"):
        self.mode = mode

    def get_transform(self):
        if self.mode == "train":
            return transforms.Compose([
                RandomCrop(500),
                Normalize(),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                CenterCrop(500),
                Normalize(),
                transforms.ToTensor(),
            ])


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        mean = np.mean(tensor)
        # mean = np.mean(tensor, axis=(0, 1))
        return (tensor - mean).astype(np.float32)