import sys

sys.path.append("./")
from src.utils.config import Config  # noqa


@Config.main
def main(cfg):
    print(Config.pretty_text(cfg))


if __name__ == "__main__":
    main()
