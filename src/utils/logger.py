import os

import torch
from torch.utils.tensorboard import SummaryWriter

from omegaconf import OmegaConf


class TensorboardLogger:
    def __init__(self, output_dir):
        self.output_dir = os.path.join(output_dir, "tensorboard")
        self.writer = SummaryWriter(self.output_dir)

    def write_model_graph(self, model, device, input_size, multi_gpu=False):
        dummy = torch.zeros(input_size, device=device)
        model.eval()
        if multi_gpu:
            self.writer.add_graph(model.module, input_to_model=[dummy])
        else:
            self.writer.add_graph(model, input_to_model=[dummy])

    def write_scalars(self, tag, value_dict, step):
        self.writer.add_scalars(tag, value_dict, step)

    def write_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def writer_close(self):
        self.writer.close()

    def writer_hparams(self, cfg):
        self.writer.add_hparams(OmegaConf.to_container(cfg, resolve=True), {})
