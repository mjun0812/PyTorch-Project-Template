import torch
from torch.multiprocessing import Manager, Value

BYTES_PER_GIB = 1024**3


class TensorCache:
    def __init__(self, size_limit_gb: int):
        self.size_limit = size_limit_gb * BYTES_PER_GIB
        self.manager = Manager()
        self.cache = self.manager.dict()
        self.current_cache_size = Value("L", 0)  # 符号なしlong型
        self.is_full = False

    def __contains__(self, key: str):
        return key in self.cache

    def _calc_elm_size(self, elm: torch.Tensor) -> int:
        size = elm.numel() * elm.dtype.itemsize
        return size

    def set(self, key: str, value: torch.Tensor) -> bool:
        if self.is_full:
            return False
        size = self._calc_elm_size(value)
        if self.current_cache_size.value + size > self.size_limit:
            self.is_full = True
            return False
        self.cache[key] = value
        self.current_cache_size.value += size
        return True

    def get(self, key: str) -> torch.Tensor | None:
        return self.cache.get(key, None)

    def clear(self):
        self.cache.clear()
        self.current_cache_size.value = 0

    def remove(self, key: str):
        if key in self.cache:
            size = self._calc_elm_size(self.cache[key])
            del self.cache[key]
            self.current_cache_size.value -= size
