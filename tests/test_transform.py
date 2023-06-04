import os
import sys
from pprint import pprint

import cv2
import hydra
import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig, OmegaConf  # noqa

sys.path.append("./")
from src.dataloaders import build_dataset  # noqa
from src.utils import box_convert_cxcywh_xyxy, box_convert_cxcywh_xyxy_np

leftkeys = (81, 110, 97, 65361, 2424832)
rightkeys = (83, 109, 100, 65363, 2555904)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    phase = "train"
    data = build_dataset(cfg, phase)
    if len(data) == 2:
        dataset, dataloader = data
        batched_transforms = None
    else:
        dataset, dataloader, batched_transforms = data
    print("Loading dataset Complete")
    device = torch.device("cpu")

    # i = 0
    # for i, _ in enumerate(dataloader):
    #     print(f"{i}: {len(dataloader)}", end="\r")
    for i, data in enumerate(dataloader):
        image, data = data
        image = image[0].unsqueeze(0).float()
        bbox = data["bbox"][0]

        box_type = cfg.MODEL.get("BOX_TYPE", "xyxy")
        if box_type == "cxcywh":
            bbox = box_convert_cxcywh_xyxy_np(bbox)
            image_size = data["image_size"][0] * data["scale"][0]
            img_w, img_h = image_size.unbind(0)
            scale_fct = np.array([img_w, img_h, img_w, img_h])
            bbox *= scale_fct
        print(image.shape, data["image_name"][0], bbox)

        image = TF.normalize(
            image,
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        image = TF.to_pil_image(image.squeeze())
        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if len(bbox) > 0:
            for box in bbox:
                cv2.rectangle(
                    image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

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
