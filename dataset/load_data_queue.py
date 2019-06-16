from collections import OrderedDict

import skimage.io as io

from config import config


class LRUCache:
    def __init__(self, capacity: int):
        self._ordered_dict = OrderedDict()
        self._capacity = capacity

    def get(self, key):
        self._move_to_end_if_exist(key)
        return self._ordered_dict.get(key)

    def put(self, key, value):
        self._move_to_end_if_exist(key)
        self._ordered_dict[key] = value
        if len(self._ordered_dict) > self._capacity:
            key, value = self._ordered_dict.popitem(last=False)
            del key
            del value

    def _move_to_end_if_exist(self, key):
        if key in self._ordered_dict:
            self._ordered_dict.move_to_end(key)


_cache = LRUCache(config["data_queue_len"])


def get_image(path):
    image = _cache.get(path)
    if image is None:
        image = io.imread(path)
        _cache.put(path, image)
    return image
