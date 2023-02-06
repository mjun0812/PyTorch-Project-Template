import sys
import cv2
import numpy as np
import os
import torch
import torchvision.transforms.functional as TF
import hydra
from omegaconf import DictConfig, OmegaConf  # noqa

sys.path.append("../")
from src.dataloaders import build_dataset  # noqa

leftkeys = (81, 110, 97, 65361, 2424832)
rightkeys = (83, 109, 100, 65363, 2555904)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    os.chdir("../")
    data = build_dataset(cfg, "train")
    if len(data) == 2:
        dataset, dataloader = data
        batched_transforms = None
    else:
        dataset, dataloader, batched_transforms = data
    print("Loading dataset Complete")
    device = torch.device("cpu")

    i = 0
    for i, data in enumerate(dataloader):
        image, info = data
        image = image[0].to(device).to(float())
        print(image.shape)
        if batched_transforms:
            image = batched_transforms(image)

        image = TF.normalize(
            image,
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        image = TF.to_pil_image(image.squeeze())
        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("Image", image)

        key = cv2.waitKeyEx(0)
        if key in rightkeys:
            i += 1
            if i >= len(dataset):
                i = 0
        if key in leftkeys:
            i -= 1
            if i < 0:
                i = len(dataset) - 1
        if (key == ord("q")) or (key == 27):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
