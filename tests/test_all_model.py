import argparse
import subprocess
from pathlib import Path


def arg_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--show", help="show all command example")
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.show:
        config_paths = sorted(list(Path("config/MODEL").glob("*/*.yaml")))
        for path in config_paths:
            print(path)

    if args.config:
        config_paths = [args.config]
    else:
        config_paths = sorted(list(Path("config/MODEL").glob("*/*.yaml")))

    for config_path in config_paths:
        command = f"./docker/run.sh python tests/test_model.py {config_path}"
        try:
            # コマンドを実行（標準出力とエラー出力をキャプチャ）
            result = subprocess.run(
                command, check=True, shell=True, text=True, capture_output=True
            )
            print(f"{command}: " + "\033[32m" + "Passed" + "\033[0m")
        except subprocess.CalledProcessError as e:
            print("\033[31mFailed\033[0m" + f": {command}")
            # print(f"Standard Output: {e.stdout}")
            # print(f"Error Output: {e.stderr}")


if __name__ == "__main__":
    main()
