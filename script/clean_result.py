import os
import sys
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.entities import ViewType

load_dotenv()


def main() -> None:
    work_dir = Path(__file__).resolve().parent.parent
    os.chdir(work_dir)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./result/mlruns")
    mlflow.set_tracking_uri(mlflow_uri)

    with open("config/__base__/config.yaml") as f:
        config = yaml.safe_load(f)
    experiment_name = config["log"]["mlflow_experiment_name"]

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    # resultディレクトリの2階層下のディレクトリの中で，config.yamlが存在するものを全て列挙する
    result_root = work_dir / "result"
    result_dirs = []
    for result_dir in result_root.glob("*/*"):
        if result_dir.is_dir():
            result_path = result_dir.relative_to(work_dir)
            if (result_path / "config.yaml").exists() or (result_path / "train.log").exists():
                result_dirs.append(str(result_path))

    # MLFlowでアクティブになっている実行結果を取得する
    runs = client.search_runs(experiment_ids=experiment_id, run_view_type=ViewType.ACTIVE_ONLY)
    if len(runs) == 0:
        print("No data")
        sys.exit(0)
    # MLFlowでアクティブになっている実行結果のoutput_dirを取得する
    mlflow_output_paths = [run.data.params["output_dir"] for run in runs]

    # resultディレクトリの2階層下のディレクトリの中で，
    # MLFlowでアクティブになっている実行結果のoutput_dirに存在しないものを列挙
    delete_dirs = []
    for result_dir in result_dirs:
        exist_in_mlflow = False
        for mlflow_output_path in mlflow_output_paths:
            if result_dir in mlflow_output_path:
                exist_in_mlflow = True
                break
        if not exist_in_mlflow:
            delete_dirs.append(result_dir)

    # 本当に削除していいかを確認
    for delete_dir in delete_dirs:
        delete_dir = Path(delete_dir)

        runs_path = delete_dir / "runs"
        if runs_path.exists():
            continue

        model_path = delete_dir / "models"
        if len(list(model_path.glob("*.pth"))) > 3:
            continue

        print(delete_dir)


if __name__ == "__main__":
    main()
