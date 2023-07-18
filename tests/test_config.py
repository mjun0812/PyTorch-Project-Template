import argparse
import sys

from addict import Dict

sys.path.append("./")
from src.utils.config import Config


@Config.main
def main(cfg):
    print(Config.pretty_text(cfg))
    print(cfg.MODEL.INPUT_SIZE, type(cfg.MODEL.INPUT_SIZE))


if __name__ == "__main__":
    main()
