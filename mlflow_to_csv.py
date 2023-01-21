import argparse
import os

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import fluent
import yaml


def arg_parse():
    parser = argparse.ArgumentParser(description="fetch result from mlflow")
    parser.add_argument("--local", help="use localhost", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = arg_parse()
    load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if args.local:
        mlflow_uri = "http://localhost:25000"
    if not mlflow_uri:
        mlflow_uri = "./result/mlruns"
    mlflow.set_tracking_uri(mlflow_uri)

    with open("config/config.yaml", "r") as f:
        experiment_name = yaml.safe_load(f)["EXPERIMENT_NAME"]
    print(f"Experiment: {experiment_name}")

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    all_runs = fluent.search_runs(experiment_ids=experiment_id)
    datasets = all_runs["params.Dataset"].unique()
    # print(datasets)
    # print(all_runs.columns)

    columns = {
        "tags.mlflow.runName": "name",
        "params.Model": "Model",
        "params.Input size": "Input size",
        "params.Loss": "Loss",
        "tags.info": "tag",
        "params.Epoch": "Epoch",
        "params.Batch": "Batch",
        "params.Optimizer": "Optimizer",
        "metrics.map_test": "mAP",
        "metrics.map_50_test": "mAP@0.5",
        "metrics.map_small_test": "mAP small",
        "metrics.map_medium_test": "mAP medium",
        "metrics.map_large_test": "mAP large",
        "params.Dataset": "Dataset",
    }
    all_runs = all_runs.rename(columns=columns)
    all_runs = all_runs[list(columns.values())]
    all_runs["name"] = all_runs["name"].apply(
        lambda x: "_".join(x.replace("-", "_").split("_")[:2])
    )

    output_dir = "doc/table"
    os.makedirs(output_dir, exist_ok=True)
    for dataset in datasets:
        idx = all_runs[all_runs["Dataset"] == dataset].index.tolist()
        runs = all_runs.loc[idx]
        runs = runs.sort_values(by=["Model", "Input size", "Loss", "name"])
        runs = runs.reset_index(drop=True)
        runs.to_csv(os.path.join(output_dir, f"{dataset}.csv"))


if __name__ == "__main__":
    main()
