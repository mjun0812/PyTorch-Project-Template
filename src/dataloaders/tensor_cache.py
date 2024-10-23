from typing import Optional

import torch
from torch.multiprocessing import Lock, Manager, Value

BYTES_PER_GIB = 1024**3


class TensorCache:
    def __init__(self, size_limit_gb: int):
        self.size_limit = size_limit_gb * BYTES_PER_GIB
        self.manager = Manager()
        self.cache = self.manager.dict()
        self.current_cache_size = Value("L", 0)  # 符号なしlong型
        self.cache_lock = Lock()
        self.max_cache_size = size_limit_gb

    def __contains__(self, key: str):
        with self.cache_lock:
            return key in self.cache

    def _calc_elm_size(self, elm: torch.Tensor) -> int:
        size = elm.numel() * elm.dtype.itemsize
        return size

    def set(self, key: str, value: torch.Tensor) -> bool:
        size = self._calc_elm_size(value)
        with self.cache_lock:
            if self.current_cache_size.value + size > self.max_cache_size:
                return False
            self.cache[key] = value
            self.current_cache_size.value += size
        return True

    def get(self, key: str) -> Optional[torch.Tensor]:
        with self.cache_lock:
            return self.cache.get(key, None)

    def clear(self):
        with self.cache_lock:
            self.cache.clear()
            self.current_cache_size.value = 0

    def remove(self, key: str):
        with self.cache_lock:
            if key in self.cache:
                size = self._calc_elm_size(self.cache[key])
                del self.cache[key]
                self.current_cache_size.value -= size
