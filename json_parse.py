import pathlib
import argparse
import json

from tabulate import tabulate


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
                with open(test_dir / "result.json", "r") as f:
                    result = json.load(f)
                data.update(result)
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
