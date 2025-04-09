from src.config import ConfigManager


@ConfigManager.argparse
def main(cfg):
    print(ConfigManager.pretty_text(cfg))


if __name__ == "__main__":
    main()
