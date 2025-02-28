import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import ConfigManager


@ConfigManager.argparse
def main(cfg):
    print(ConfigManager.pretty_text(cfg))


if __name__ == "__main__":
    main()
