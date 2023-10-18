import argparse
import os
import sys
import tempfile

import mlflow
import yaml
from dotenv import load_dotenv
from natsort import natsorted
from tabulate import tabulate

os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


def arg_parse():
    parser = argparse.ArgumentParser(description="fetch result from mlflow")
    parser.add_argument("dataset", help="Dataset Name or all")
    parser.add_argument(
        "--format",
        help="table format. Default value is 'simple'",
        type=str,
        default="github",
        choices=["simple", "plain", "html", "latex", "latex_row", "github"],
    )
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

    with open("config/__BASE__/config.yaml", "r") as f:
        experiment_name = yaml.safe_load(f)["EXPERIMENT_NAME"]
    print(f"Experiment: {experiment_name}")

    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string=f"params.Dataset LIKE '%{args.dataset}%'" if args.dataset != "all" else "",
    )
    if len(runs) == 0:
        print("No data")
        sys.exit(0)

    os.makedirs("./.tmp", exist_ok=True)

    table_data = []
    for run in runs:
        model = run.data.params.get("Model", "")
        name = "_".join(run.data.tags["mlflow.runName"].split("_")[:2])
        loss = run.data.params["Loss"]

        config = None
        backbone_name = None
        dataset_name = None
        tag = None

        if run.info.status in ["FINISHED", "FAILED"] and os.path.isfile(
            f"./.tmp/{run.data.tags['mlflow.runName']}.yaml"
        ):
            with open(f"./.tmp/{run.data.tags['mlflow.runName']}.yaml", "r") as f:
                config = yaml.safe_load(f)
        else:
            artifacts = client.list_artifacts(run_id=run.info.run_id)
            for a in artifacts:
                if a.path.endswith("config.yaml"):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        mlflow.artifacts.download_artifacts(
                            run_id=run.info.run_id, artifact_path=a.path, dst_path=tmpdir
                        )
                        with open(os.path.join(tmpdir, "config.yaml"), "r") as f:
                            config = yaml.safe_load(f)
                        with open(f"./.tmp/{run.data.tags['mlflow.runName']}.yaml", "w") as f:
                            yaml.dump(config, f)
                    break

        if config:
            backbone_name = config["MODEL"].get("BACKBONE", None)
            dataset_name = config["DATASET"].get("NAME", None)
            tag = config.get("TAG", None)

        input_size = run.data.params["Input size"]

        mlflow_metrics = run.data.metrics
        mean_ap = mlflow_metrics.get("map_test")
        mean_ap_50 = mlflow_metrics.get("map_50_test")
        if mean_ap_50 is None:
            mean_ap_50 = mlflow_metrics.get("AP/IoU 0.50_test")
        # fps = mlflow_metrics.get("fps_test")

        table_data.append(
            {
                "Name": name,
                "Dataset": dataset_name,
                "Backbone": backbone_name,
                "Model": model,
                "Loss": loss,
                "Input size": input_size,
                "mAP": mean_ap,
                "mAP 0.5": mean_ap_50,
                "Tag": tag,
                # "FPS": fps,
            }
        )
    table = tabulate(
        natsorted(table_data, key=lambda x: f"{x['Model']}_{x['Loss']}"),
        headers="keys",
        tablefmt=args.format,
    )
    if args.format == "latex":
        table = "\\begin{table}[htbp]\n\\centering\n\\caption{}\n" + table
        table += "\\end{table}"
    print(table)

    # table = tabulate(
    #     natsorted(table_data, key=lambda x: f"{x['Model']}_{x['Loss']}"),
    #     headers="keys",
    #     tablefmt="github",
    # )
    # with open(f"./doc/{args.dataset}.md", "w") as f:
    #     f.write(table)


if __name__ == "__main__":
    main()
