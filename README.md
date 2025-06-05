# PyTorch Project Template

My project template using PyTorch.

## Features

- Docker + uv environment setup.
- Single-node or multi-node and multi-GPU training on DDP.
- Experiment management with MLflow and wandb.
- Config management with OmegaConf and dataclasses.
- Resume training support.
- Supports development on macOS and Linux.
- Devcontainer support.
- CI support with pre-commit and GitHub Actions.

## Environments

- Python 3.11
- CUDA 12.8
- PyTorch 2.7.1

## Install

Create env file.

```bash
cp template.env .env
vim .env
```

```bash
# Notice to Slack when finished trianing and evaluation.
SLACK_TOKEN="HOGE"

# Write local path or remote uri
MLFLOW_TRACKING_URI="./result/mlruns"
# Basic Auth
# MLFLOW_TRACKING_USERNAME=""
# MLFLOW_TRACKING_PASSWORD=""

WANDB_API_KEY=""
```

- Docker Install

```bash
./docker/build.sh
./docker/run.sh [command]
```

- Local Install

```bash
uv sync
uv run [command]
```

- Build develop environment

Use devcontainer or local development.

```bash
uv sync
uv run pre-commit install
```

## Usage

### Tools

```bash
# JupyterLab Server
./script/run_notebook.sh

# MLflow WebUI
./script/run_mlflow.sh

# Replace Config value
python script/edit_configs.py [yaml or dir] "params.hoge=aa,params.fuga=bb"

# Show deleted local result from mlflow
# Delete experiment results from the local `./result/` directory that have been removed from MLflow.
python script/clean_result.py | xargs -I{} -P 2 rm -rf {}

# Aggregate mlflow result
python script/aggregate_mlflow.py [dataset_name or all]

# Check Registered Modules
python script/show_registers.py
```

### Config management

This template uses OmegaConf and dataclasses for config management.

You can change the Config values in the yaml file from the CLI.
Use `.` to concatenate and `=` to specify the value.

```yaml
gpu:
  use: 1
```

```bash
python train.py config/model/ResNet.yaml gpu.use=2
```

### Training

To perform training, use `train.py`. This script also runs the auto test after training.
In `train.py`, specify the yaml file under `config`.

```bash
python train.py [config_file_path]
# Example
python train.py config/model/ResNet.yaml batch=8
```

Training results are saved under the directory `result/[train_dataset.name]/[date]_[model.name]_[tag]`.

#### Single Node Multi GPU Training

To perform training with a single node and multiple GPUs,
remove `python` from the command and add `./torchrun.sh [number of GPUs]` before it,
and change the Config value like `gpu.use="0,1"`.
At this time, the order of GPU IDs is the order of PCIe as shown by the `nvidia-smi` command.

```bash
./torchrun.sh 4 train.py config/model/ResNet.yaml gpu.use="0,1,2,3"
```

#### Multi Node Multi GPU Training

To perform training with multiple nodes and multiple GPUs,
remove `python` from the command and add `./multinode.sh [number of nodes] [number of GPUs] [job ID] [node rank] [master node hostname:master node port]` before it,
and change the Config value like `gpu.use="0,1"`.

```bash
# Master Node
./multinode.sh 2 4 12345 0 localhost:12345 train.py config/model/ResNet.yaml gpu.use=0,1,2,3

# Worker Node
./multinode.sh 2 4 12345 1 192.168.1.10:12345 train.py config/model/ResNet.yaml gpu.use=4,5,6,7
```

#### RAM Cache

There is a function to cache part of the dataset in RAM. The cache only supports `torch.Tensor`.

```bash
python train.py config/model/ResNet.yaml gpu.use=1 use_ram_cache=true ram_cache_size_gb=16
```

To use this function, you need to implement the dataset as follows.

```python
if self.cache is not None and idx in self.cache:
    image = self.cache.get(idx)
else:
    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    if self.cache is not None:
        self.cache.set(idx, image)
```

#### Resume Training

After training is completed or interrupted,
you can resume training by specifying the `config.yaml` saved in the result directory,
`result/[train_dataset.name]/[date]_[model.name]_[tag]/config.yaml`.
At this time, even if the original epoch was 100, if you specify `epoch=150` on the command line, the config will be overwritten and training will continue until 150 epochs.

```bash
python train.py config/config.yaml # Completed up to 100 epochs

# Resume or continue training using the above results
python train.py result/ImageNet/hoge_hoge/config.yaml epoch=150 gpu.use=7 # Continue training until 150 epochs
```

### Evaluate

The script for evaluation is `test.py`.
Evaluation is also executed in `train.py` for training, but use this if you want to do it manually.
In `test.py`, specify the `config.yaml` in the directory where the training results are saved as the first argument.

```bash
python test.py result/ImageNet/hoge_hoge/config.yaml gpu.use=7
```
