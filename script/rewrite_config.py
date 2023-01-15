import os
import re

import yaml


def main():
    result_dirs = []
    for dir, _, _ in os.walk("../result"):
        depth = dir.count(os.path.sep)
        if depth >= 4 or depth == 0:
            continue
        if not re.match("^20", os.path.basename(dir)):
            continue
        result_dirs.append(dir)
        if os.path.isdir(os.path.join(dir, "runs")):
            print(dir)
            with open(os.path.join(dir, "config.yaml"), "r") as f:
                cfg = yaml.safe_load(f)

            cfg["DATASET"]["TYPE"] = rewrite(cfg["DATASET"]["TYPE"])

            with open(os.path.join(dir, "config.yaml"), "w") as f:
                yaml.dump(cfg, f)


def rewrite(cfg):
    if cfg == "CSV":
        return "CsvDataset"
    elif cfg == "COCO":
        return "CocoDataset"
    return cfg


if __name__ == "__main__":
    main()
