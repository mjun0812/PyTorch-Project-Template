import os
import re
import shutil
import sys

import mlflow
import yaml
from dotenv import load_dotenv


def main():
    load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    # mlflow_uri = "http://localhost:25000"
    if not mlflow_uri:
        mlflow_uri = "./result/mlruns"
    mlflow.set_tracking_uri(mlflow_uri)

    with open("config/__BASE__/config.yaml", "r") as f:
        experiment_name = yaml.safe_load(f)["MLFLOW_EXPERIMENT_NAME"]
    # print(f"Experiment: {experiment_name}")

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    runs = client.search_runs(experiment_ids=experiment_id)
    if len(runs) == 0:
        print("No data")
        sys.exit(0)

    # get result
    mlflow_dirs = []
    for run in runs:
        name = run.data.tags["mlflow.runName"]
        mlflow_dirs.append(name)

    result_dirs = []
    for dir, _, _ in os.walk("result"):
        depth = dir.count(os.path.sep)
        if depth >= 3 or depth == 0:
            continue
        if not re.match("^20", os.path.basename(dir)):
            continue
        result_dirs.append(dir)

    for dir in result_dirs:
        if os.path.basename(dir) not in mlflow_dirs:
            print(dir)
            # shutil.rmtree(dir)


if __name__ == "__main__":
    main()
