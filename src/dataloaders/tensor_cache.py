import torch
from torch.multiprocessing import Manager, Value

BYTES_PER_GIB = 1024**3
"""Number of bytes per gibibyte (1024^3)."""


class TensorCache:
    """Thread-safe tensor cache with size limits.

    Provides a shared memory cache for tensors to improve data loading
    performance across multiple processes.

    Attributes:
        size_limit: Maximum cache size in bytes.
        manager: Multiprocessing manager for shared data structures.
        cache: Shared dictionary storing cached tensors.
        current_cache_size: Current cache size in bytes.
        is_full: Whether the cache has reached its size limit.
    """

    def __init__(self, size_limit_gb: int) -> None:
        """Initialize the tensor cache.

        Args:
            size_limit_gb: Maximum cache size in gigabytes.
        """
        self.size_limit = size_limit_gb * BYTES_PER_GIB
        self.manager = Manager()
        self.cache = self.manager.dict()
        self.current_cache_size = Value("L", 0)  # unsigned long type
        self.is_full = False

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key to check.

        Returns:
            True if the key exists in the cache.
        """
        return key in self.cache

    def _calc_elm_size(self, elm: torch.Tensor) -> int:
        """Calculate the size of a tensor in bytes.

        Args:
            elm: Tensor to calculate size for.

        Returns:
            Size of the tensor in bytes.
        """
        size = elm.numel() * elm.dtype.itemsize
        return size

    def set(self, key: str, value: torch.Tensor) -> bool:
        """Store a tensor in the cache.

        Args:
            key: Cache key for the tensor.
            value: Tensor to store.

        Returns:
            True if the tensor was successfully cached, False if cache is full.
        """
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
        """Retrieve a tensor from the cache.

        Args:
            key: Cache key for the tensor.

        Returns:
            Cached tensor if found, None otherwise.
        """
        return self.cache.get(key, None)

    def clear(self) -> None:
        """Clear all cached tensors and reset cache size."""
        self.cache.clear()
        self.current_cache_size.value = 0

    def remove(self, key: str) -> None:
        """Remove a specific tensor from the cache.

        Args:
            key: Cache key for the tensor to remove.
        """
        if key in self.cache:
            size = self._calc_elm_size(self.cache[key])
            del self.cache[key]
            self.current_cache_size.value -= size
