import pathlib
import re
import argparse
import pandas as pd

SEARCH_TEXT = ["Accuracy", "Speed"]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def parse_log(result_dir):
    directories = pathlib.Path(result_dir).glob("**/runs/?*/")
    names = []
    results = []
    non_exist = []
    for d in directories:
        if not d.is_dir():
            continue
        name = [d.parents[1].name, d.name]
        log = d / "test.log"
        try:
            with open(log, "r") as f:
                log_txt = f.readlines()
            result = []
            for text in SEARCH_TEXT:
                for row in reversed(log_txt):
                    if text in row:
                        # if text ==
                        if text == "Speed":
                            row = row.split(",")[-1]
                        result.append(row)
                        break

            result = list(map(lambda x: x.split(":")[-1].strip(), result))
            if len(result) == 0:
                raise ValueError
            row = [*name, *result]
            names.append("/".join(name))
            results.append(row)
        except Exception:
            non_exist.append("/".join(name))
            continue

    pd.options.display.colheader_justify = "left"
    pd.set_option("display.width", None)
    df = (
        pd.DataFrame(data=results, columns=["NAME", "RUNS", *SEARCH_TEXT])
        .sort_values(["NAME", "RUNS"])
        .reset_index(drop=True)
    )
    max_name = len(max(df["NAME"], key=len))
    max_runs = len(max(df["RUNS"], key=len))

    df["NAME"] = df["NAME"].map(f"{{:<{max_name}}}".format)
    df["RUNS"] = df["RUNS"].map(f"{{:<{max_runs}}}".format)
    print(df)
    if len(non_exist) > 0:
        print("non_exist:", non_exist)


def arg_parse():
    parser = argparse.ArgumentParser(description="parse test_log.txt")
    parser.add_argument("result", help="result_dir")
    return parser.parse_args()


def main():
    args = arg_parse()
    parse_log(args.result)


if __name__ == "__main__":
    main()
