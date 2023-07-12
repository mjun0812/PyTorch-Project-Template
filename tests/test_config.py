import sys

sys.path.append("./")
from src.utils.config import Config, auto_argparser

cfg = auto_argparser()

print(Config.pretty_text(cfg))
