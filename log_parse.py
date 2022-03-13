import pathlib
import argparse

from tabulate import tabulate

SEARCH_TEXT = ["Accuracy", "Speed"]


def parse_log(base_dir, format="simple"):
    test_dirs = pathlib.Path(base_dir).glob("**/runs/?*/")
    table_data = []
    parse_error = []

    for d in test_dirs:
        if not d.is_dir():
            continue

        data = {}
        data["name"] = d.parents[1].name
        data["runs"] = d.name

        try:
            with open(d / "test.log", "r") as f:
                log_txt = f.readlines()
            for keyword in SEARCH_TEXT:
                for row in reversed(log_txt):
                    if keyword in row:
                        # [2021-11-26 07:51:17,843][INFO] AP@IoU=0.75: 0.99500 -> AP@IoU=0.75: 0.99500
                        metrics = row.strip().split("]")[-1]
                        # AP@IoU=0.75: 0.99500 -> 0.99500
                        metrics = metrics.split(":")[-1].strip()

                        ####### If you use specific log, edit code below. #######
                        # if keyword == "Speed":
                        #     # Average Inferance Speed: 0.02623s, 38.12fps

                        data[keyword] = metrics
                        break
            if len(data.keys()) > 2:
                table_data.append(data)
            else:
                raise KeyError
        except Exception:
            parse_error.append(data["name"])
            continue

    print(tabulate(table_data, headers="keys", tablefmt=format))

    if len(parse_error) > 0:
        print("parse_error:", parse_error)


def arg_parse():
    parser = argparse.ArgumentParser(description="parse test_log.txt")
    parser.add_argument("result", help="result_dir")
    parser.add_argument(
        "--format",
        help="table format. Default value is 'simple'",
        type=str,
        default="simple",
        choices=["simple", "plain", "html", "latex", "latex_row", "github"],
    )
    return parser.parse_args()


def main():
    args = arg_parse()
    parse_log(args.result, format=args.format)


if __name__ == "__main__":
    main()
