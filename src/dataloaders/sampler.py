from torch.utils.data import BatchSampler

from ..utils import Registry

SAMPLER_REGISTRY = Registry("SAMPLER")


@SAMPLER_REGISTRY.register()
class BatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        return super().__iter__()
