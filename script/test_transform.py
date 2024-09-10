import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import ConfigManager, ExperimentConfig
from src.dataloaders import build_dataset

leftkeys = (81, 110, 97, 65361, 2424832)
rightkeys = (83, 109, 100, 65363, 2555904)


@ConfigManager.argparse
def main(cfg: ExperimentConfig):
    cfg.batch = 2
    phase = "train"
    data = build_dataset(cfg, phase)
    _, dataloader, batched_transforms = data

    denormalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    for _, data in enumerate(dataloader):
        if batched_transforms:
            data = batched_transforms(data)
        print(f"Data: {data.keys()}")
        if "image" not in data:
            continue

        image = data["image"][0]
        print(f"Image Shape: {image.shape}")

        image = denormalize(image)
        image = image.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if "bbox" in data:
            bbox = data["bbox"][0].numpy()
            for box in bbox:
                cv2.rectangle(
                    image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
        if "label" in data and data["label"].dim() > 3:
            mask = TF.to_pil_image(data["label"].squeeze().type(torch.uint8))
            palette = data["color_palette"][0]
            mask.putpalette(list(palette))
            mask = np.array(mask.convert("RGB"), dtype=np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            cv2.imshow("Mask", mask)

        cv2.imshow("Image", image)

        key = cv2.waitKeyEx(0)
        if (key == ord("q")) or (key == 27):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
