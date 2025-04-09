import os
import sys
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.entities import ViewType

load_dotenv()


def main():
    work_dir = Path(__file__).resolve().parent.parent
    os.chdir(work_dir)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./result/mlruns")
    mlflow.set_tracking_uri(mlflow_uri)

    with open("config/__base__/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    experiment_name = config["log"]["mlflow_experiment_name"]

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    runs = client.search_runs(experiment_ids=experiment_id, run_view_type=ViewType.DELETED_ONLY)
    if len(runs) == 0:
        print("No data")
        sys.exit(0)

    # get result
    for run in runs:
        output_dir = run.data.params["output_dir"]
        if Path(output_dir).exists():
            print(output_dir)


if __name__ == "__main__":
    main()
