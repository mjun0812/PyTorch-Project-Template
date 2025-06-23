import time
from collections.abc import Iterator

from torch.utils.data import DataLoader

from .types import DatasetOutput


class IterBasedDataloader:
    """Iteration-based dataloader for training loops.

    Provides a dataloader that yields batches for a fixed number of iterations
    and automatically handles epoch transitions.

    Attributes:
        max_iter: Maximum number of iterations.
        step_iter: Number of iterations per epoch.
        dataloader: Underlying PyTorch DataLoader.
        iterator: Infinite iterator over the dataloader.
        current_iter: Current iteration count within the epoch.
        epoch: Current epoch number.
        batch_size: Batch size from the underlying dataloader.
        sampler: Sampler from the underlying dataloader.
    """

    def __init__(self, dataloader: DataLoader, max_iter: int, step_iter: int) -> None:
        """Initialize the iteration-based dataloader.

        Args:
            dataloader: PyTorch DataLoader to wrap.
            max_iter: Maximum number of iterations to run.
            step_iter: Number of iterations per epoch.
        """
        self.max_iter = max_iter
        self.step_iter = step_iter
        self.dataloader = dataloader
        self.iterator = InfiniteDataLoader(self.dataloader)
        self.current_iter = 0
        self.epoch = 0
        self.batch_size = self.dataloader.batch_size
        self.sampler = self.dataloader.sampler

    def __len__(self) -> int:
        """Return the number of iterations per epoch.

        Returns:
            Number of iterations per epoch.
        """
        return self.step_iter

    def __iter__(self) -> Iterator[DatasetOutput]:
        """Return iterator over the dataloader.

        Returns:
            Iterator yielding dataset outputs.
        """
        return self

    def __next__(self) -> DatasetOutput:
        """Get the next batch from the dataloader.

        Returns:
            Next dataset output.

        Raises:
            StopIteration: When max iterations reached or epoch completed.
        """
        if self.current_iter + self.step_iter * self.epoch > self.max_iter:
            raise StopIteration
        if self.current_iter != 0 and self.current_iter % self.step_iter == 0:
            self.epoch += 1
            self.current_iter = 0
            raise StopIteration
        self.current_iter += 1

        data = next(self.iterator)
        return data


class InfiniteDataLoader:
    """Infinite dataloader that cycles through data indefinitely.

    Automatically restarts iteration when the dataloader is exhausted,
    handling epoch transitions for distributed training.

    Attributes:
        dataloader: Underlying PyTorch DataLoader.
        iterator: Current iterator over the dataloader.
        epoch: Current epoch number.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        """Initialize the infinite dataloader.

        Args:
            dataloader: PyTorch DataLoader to wrap.
        """
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.epoch = 0

    def __iter__(self) -> Iterator[DatasetOutput]:
        """Return iterator over the dataloader.

        Returns:
            Iterator yielding dataset outputs.
        """
        return self

    def __next__(self) -> DatasetOutput:
        """Get the next batch from the dataloader.

        Automatically handles epoch transitions and sampler state updates
        for distributed training.

        Returns:
            Next dataset output.
        """
        try:
            data = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if hasattr(self.dataloader, "sampler") and hasattr(
                self.dataloader.sampler, "set_epoch"
            ):
                self.dataloader.sampler.set_epoch(self.epoch)
            elif hasattr(self.dataloader, "batch_sampler") and hasattr(
                self.dataloader.batch_sampler, "set_epoch"
            ):
                self.dataloader.batch_sampler.sampler.set_epoch(self.epoch)

            time.sleep(1)  # Prevent possible deadlock during epoch transition
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data
