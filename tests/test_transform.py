import sys

import cv2
import numpy as np
import torchvision.transforms.functional as TF

sys.path.append("./")
from src.dataloaders import build_dataset  # noqa
from src.utils import Config  # noqa

leftkeys = (81, 110, 97, 65361, 2424832)
rightkeys = (83, 109, 100, 65363, 2555904)


@Config.main
def main(cfg):
    phase = "train"
    cfg.BATCH = 1
    cfg.CPU = True
    data = build_dataset(cfg, phase)
    _, dataloader, batched_transforms = data
    print("Loading dataset Complete")

    for _, data in enumerate(dataloader):
        image, data = data
        if batched_transforms:
            image, _ = batched_transforms(image, data)

        print(image.shape)

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
        if (key == ord("q")) or (key == 27):
            cv2.destroyAllWindows()
            break
        else:
            continue


if __name__ == "__main__":
    main()
