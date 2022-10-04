import os
import sys
import re
import shutil

from dotenv import load_dotenv
import mlflow
import yaml


def main():
    load_dotenv(dotenv_path=f"{os.environ['HOME']}/.env")
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    # mlflow_uri = "http://localhost:25000"
    if not mlflow_uri:
        mlflow_uri = "./result/mlruns"
    mlflow.set_tracking_uri(mlflow_uri)

    with open("config/config.yaml", "r") as f:
        experiment_name = yaml.safe_load(f)["EXPERIMENT_NAME"]
    print(f"Experiment: {experiment_name}")

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    runs = client.search_runs(experiment_ids=experiment_id)
    if len(runs) == 0:
        print("No data")
        sys.exit(0)

    # get result
    mlflow_dirs = []
    for run in runs:
        name = run.data.tags["mlflow.runName"].replace("-", "_")
        mlflow_dirs.append(name)

        test_weight = run.data.tags.get("model_weight_test")
        if test_weight:
            test_weight = os.path.basename(os.path.dirname(os.path.dirname(test_weight)))
            mlflow_dirs.append(test_weight)

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
            print("remove", dir)
            # shutil.rmtree(dir)


if __name__ == "__main__":
    main()
