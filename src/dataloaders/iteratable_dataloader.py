import time

from torch.utils.data import DataLoader


class IterBasedDataloader:
    def __init__(self, dataloader: DataLoader, max_iter: int, step_iter: int) -> None:
        self.max_iter = max_iter
        self.step_iter = step_iter
        self.dataloader = dataloader
        self.iterator = InfiniteDataLoader(self.dataloader)
        self.current_iter = 0
        self.epoch = 0
        self.batch_size = self.dataloader.batch_size
        self.sampler = self.dataloader.sampler

    def __len__(self):
        return self.step_iter

    def __iter__(self):
        return self

    def __next__(self):
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
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
        self.epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
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
