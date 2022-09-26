import argparse
import os
import sys

from dotenv import load_dotenv
from tabulate import tabulate
import mlflow
import yaml


def arg_parse():
    parser = argparse.ArgumentParser(description="fetch result from mlflow")
    parser.add_argument("dataset", help="Dataset Name")
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
    load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        mlflow_uri = "./result/mlruns"
    mlflow.set_tracking_uri(mlflow_uri)

    with open("config/config.yaml", "r") as f:
        experiment_name = yaml.safe_load(f)["EXPERIMENT_NAME"]
    print(f"Experiment: {experiment_name}")

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    runs = client.search_runs(
        experiment_ids=experiment_id, filter_string=f"params.Dataset = '{args.dataset}'"
    )
    if len(runs) == 0:
        print("No data")
        sys.exit(0)

    table_data = []
    for run in runs:
        name = run.data.tags["mlflow.runName"]
        model = run.data.params["Model"]
        loss = run.data.params["Loss"]
        input_size = run.data.params["Input size"]

        mlflow_metrics = run.data.metrics
        mean_ap = mlflow_metrics.get("map_test")

        mean_ap_50 = mlflow_metrics.get("map_50_test")
        if mean_ap_50 is None:
            mean_ap_50 = mlflow_metrics.get("AP/IoU 0.50_test")

        fps = mlflow_metrics.get("fps_test")

        table_data.append(
            {
                "Name": name,
                "Model": model,
                "Loss": loss,
                "Input size": input_size,
                "mAP": mean_ap,
                "mAP 0.5": mean_ap_50,
                "FPS": fps,
            }
        )
    table = tabulate(
        sorted(table_data, key=lambda x: x["Model"]), headers="keys", tablefmt=args.format
    )
    if args.format == "latex":
        table = "\\begin{table}[htbp]\n\\centering\n\\caption{}\n" + table
        table += "\\end{table}"
    print(table)


if __name__ == "__main__":
    main()
