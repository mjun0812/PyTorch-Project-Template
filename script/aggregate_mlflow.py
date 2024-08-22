import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv

os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
load_dotenv()

TEMP_DIR = "./.tmp"


def arg_parse():
    parser = argparse.ArgumentParser(description="fetch result from mlflow")
    parser.add_argument("dataset", help="Dataset Name or all")
    parser.add_argument(
        "--running", "-r", help="show running experiments", action="store_true", default=False
    )
    return parser.parse_args()


def get_artifact_list_recursive(
    client: mlflow.tracking.MlflowClient, run: mlflow.entities.Run, path=""
):
    run_id = run.info.run_id
    artifacts = client.list_artifacts(run_id=run_id, path=path)
    artifact_list = []
    for a in artifacts:
        if a.is_dir:
            artifact_list.extend(get_artifact_list_recursive(client, run, a.path))
        else:
            artifact_list.append(a)
    return artifact_list


def get_artifacts_from_mlflow(client: mlflow.tracking.MlflowClient, run: mlflow.entities.Run):
    run_id = run.info.run_id
    run_name = run.data.tags["mlflow.runName"]
    artifacts = get_artifact_list_recursive(client, run)
    config = None
    for a in artifacts:
        if a.is_dir:
            continue
        elif a.path.endswith("config.yaml"):
            with tempfile.TemporaryDirectory() as d:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=a.path, dst_path=d
                )
                with open(local_path, "r") as f:
                    config = yaml.safe_load(f)
                config["mlflow_status"] = run.info.status
                with open(f"./.tmp/{run_name}_config.yaml", "w") as f:
                    yaml.dump(config, f, sort_keys=False)
        elif a.path.endswith("result.json"):
            with tempfile.TemporaryDirectory() as d:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=a.path, dst_path=d
                )
                shutil.move(local_path, f"./.tmp/{run_name}_result.json")
    return config, f"./.tmp/{run.data.tags['mlflow.runName']}_result.json"


def main():
    args = arg_parse()
    Path(TEMP_DIR).mkdir(exist_ok=True, parents=True)
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./result/mlruns")
    mlflow.set_tracking_uri(mlflow_uri)

    with open("config/__base__/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    experiment_name = cfg["mlflow"]["experiment_name"]
    print(f"Experiment: {experiment_name}")

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    options = dict(experiment_ids=experiment_id, filter_string="")
    if args.dataset != "all":
        options["filter_string"] = f"params.Train_Dataset LIKE '%{args.dataset}%'"
    if not args.running:
        if options["filter_string"]:
            options["filter_string"] = "(" + options["filter_string"] + ") AND status='FINISHED'"
        else:
            options["filter_string"] = "status='FINISHED'"
    runs = client.search_runs(**options)

    if len(runs) == 0:
        print("No data")
        sys.exit(0)

    table_data = []
    for run in runs:
        name = "_".join(run.data.tags["mlflow.runName"].split("_")[:2])  # 日付だけ
        model = run.data.params.get("Model", "")
        result = dict(name=name, model=model)

        local_config_path = f"./.tmp/{run.data.tags['mlflow.runName']}_config.yaml"
        local_result_path = f"./.tmp/{run.data.tags['mlflow.runName']}_result.json"

        # load config from mlflow artifact or cache
        if os.path.isfile(local_config_path) and os.path.isfile(local_result_path):
            with open(local_config_path, "r") as f:
                config = yaml.safe_load(f)
            if run.info.status != config["mlflow_status"]:
                config, local_result_path = get_artifacts_from_mlflow(client, run)
        else:
            config, local_result_path = get_artifacts_from_mlflow(client, run)
        with open(local_result_path, "r") as f:
            result.update(json.load(f))
        if config is None:
            continue

        table_data.append(result)

    df = pd.DataFrame(table_data)
    print(df.to_markdown(tablefmt=args.format))

    Path("./doc/result_csv").mkdir(exist_ok=True, parents=True)
    df.to_csv(
        f"./doc/result_csv/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.dataset}_result.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
