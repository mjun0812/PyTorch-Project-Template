import os

import torch
from torch.multiprocessing import Manager, Value

BYTES_PER_GIB = 1024**3


class TensorCache:
    def __init__(self, size_limit_gb):
        self.size_limit_gb = size_limit_gb
        self.manager = Manager()
        self.cache = self.manager.dict()

        self.current_cache_size = Value("d", 0)
        self.max_cache_size = size_limit_gb * 0.9

    def __contains__(self, key: str):
        return key in self.cache

    def _calc_elm_size(self, elm):
        size = elm.numel() * elm.dtype.itemsize
        return size

    def set(self, key: str, value: torch.Tensor):
        size = self._calc_elm_size(value) / BYTES_PER_GIB
        if self.current_cache_size.value + size > self.max_cache_size:
            return
        self.cache[key] = value
        self.current_cache_size.value += size

    def get(self, key: str):
        if key not in self.cache:
            return None
        return self.cache[key]

    def clear(self):
        self.cache.clear()
        self.current_cache_size = 0


def get_shm_size() -> int:
    stats = os.statvfs("/dev/shm")
    shm_bytes = stats.f_bsize * stats.f_blocks
    return shm_bytes
