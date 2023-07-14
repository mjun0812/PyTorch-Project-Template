import argparse
import sys

from addict import Dict

sys.path.append("./")
from src.utils.config import Config

# cfg = Config.build()

# print(Config.pretty_text(cfg))


@Config.main
def main(cfg):
    print(Config.pretty_text(cfg))


main()
