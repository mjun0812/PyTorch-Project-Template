import argparse
import pathlib

from tabulate import tabulate

SEARCH_TEXT = ["Accuracy", "Speed"]


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


def parse_log(base_dir, format="simple"):
    table_data = []
    parse_error = []
    no_test_runs = []

    for train_dir in pathlib.Path(base_dir).iterdir():
        test_dirs = list(train_dir.glob("runs/?*/"))

        if len(test_dirs) == 0:
            no_test_runs.append(train_dir.name)
            continue

        for test_dir in test_dirs:
            test_dir = pathlib.Path(test_dir)
            data = {}
            data["name"] = test_dir.parents[1].name
            data["runs"] = test_dir.name

            try:
                with open(test_dir / "test.log", "r") as f:
                    log_txt = f.readlines()
                for keyword in SEARCH_TEXT:
                    for row in reversed(log_txt):
                        if keyword in row:
                            # [2021-11-26 07:51:17,843][INFO] AP@IoU=0.75: 0.99500 -> AP@IoU=0.75: 0.99500
                            metrics = row.strip().split("]")[-1]
                            # AP@IoU=0.75: 0.99500 -> 0.99500
                            metrics = metrics.split(":")[-1].strip()

                            # ###### If you use specific log, edit code below. #######
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

    table = tabulate(table_data, headers="keys", tablefmt=format)
    if format == "latex":
        table = "\\begin{table}[htbp]\n\\centering\n\\caption{}\n" + table
        table += "\\end{table}"
    print(table)

    if len(no_test_runs) > 0:
        print("no test runs:", no_test_runs)
    if len(parse_error) > 0:
        print("parse_error:", parse_error)


def main():
    args = arg_parse()
    parse_log(args.result, format=args.format)


if __name__ == "__main__":
    main()
