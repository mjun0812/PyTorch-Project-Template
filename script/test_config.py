import sys

sys.path.append("./")
from src.config import ConfigManager


@ConfigManager.argparse
def main(cfg):
    print(ConfigManager.pretty_text(cfg))
    # ConfigManager.dump(cfg, "./test_config.yaml")


if __name__ == "__main__":
    main()
