# PyTorch Project Template

My PyTorch Project Template.

## Environments

This repository provided Dockerfile. But you can execute in local environment.

- CUDA >= 11.0
- Python >= 3.9
- PyTorch >= 1.11.0, != 1.12.0
- [kunai](https://github.com/mjun0812/kunai) (My Python Package)

## Features

- Provided Docker environment
- Distributed Multi GPU Training
- Mix Precision(`torch.amp`) Training
- Continue Training from your own weight
- Early Stopping
- [MLflow](https://mlflow.org) manage experiments
- Tensorboard Logging

## Install

`pip install -r requirements.txt`

### Optional: MLflow

Set up MLflow Tracking at local storage or remote server.

```bash
$ cp template.env .env
$ vim .env

SLACK_TOKEN="HOGE"
MLFLOW_TRACKING_URI=""
# MLFLOW_TRACKING_USERNAME=""
# MLFLOW_TRACKING_PASSWORD=""
```

If `MLFLOW_TRACKING_URI` is blank(""), MLflow files is saved local directry.  
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` is Basic Authentication parameters for remote MLflow Tracing server.  
`USE_MLFLOW` is default `True`.

```bash
python train.py config/MODEL/model.yaml --USE_MLFLOW True --EXPERIMENT_NAME "pytorch-experiment"
```

## Usage

```bash
./docker/build.sh # Build Docker Image from ./docker/Dockerfile
./docker/run.sh python train.py config/MODEL/model.yaml --GPU.USE 1
./docker/run.sh python test.py result/20220911/config.yaml --GPU.USE 1
```

### Distributed Multi GPU Training

Distributed Multi GPU Training is implemented in this repository using `torchrun`.  
Change Setting in `config/config.yaml`(below).

```yaml
# Device Params
GPU:
  USE: 0
  MULTI: False
CPU: False
```

If you use multi GPU, set `GPU.MULTI: True` and `GPU.USE: "0,1,2,3"`.  
This setting can change from CLI using Hydra.

```bash
# Single node, 2 GPUs
./torchrun.sh 2 train.py config/MODEL/model.yaml --GPU.MULTI True --GPU.USE '1,2'
./docker/run.sh ./torchrun.sh 2 train.py config/MODEL/model.yaml --GPU.MULTI True --GPU.USE '1,2'
```
