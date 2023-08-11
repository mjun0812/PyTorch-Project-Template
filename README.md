# PyTorch Project Template

My PyTorch Project Template.

## Environments

This repository provided Dockerfile. But you can execute in local environment.

- CUDA >= 11.0
- Python 3.11
- Poetry
- PyTorch 2.0.0, != 1.12.0

## Features

- Poetry + Docker build environment
- Multi GPU Training (PyTorch DDP or DP)
- Mix Precision(`torch.amp`) Training
- [MLflow](https://mlflow.org) manage experiments
- [OmegaConf](https://github.com/omry/omegaconf) config managements

## Install

```bash
./docker/build.sh
ln -sfv "[datasets dir]" ./dataset
mkdir result
./docker/run.sh python ...
```

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

If `MLFLOW_TRACKING_URI` is blank(""), MLflow dir is saved local directry(`result/mlruns`).  
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

### Multi GPU Training

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
This setting can change from CLI.

```bash
# DDP Single node, 2 GPUs
./docker/run.sh ./torchrun.sh 2 train.py config/MODEL/model.yaml --GPU.USE '1,2'
```

The above is implemented with DDP, but it also works with DP with the following command

```bash
# DP
./docker/run.sh python train.py config/MODEL/model.yaml --GPU.USE "0,1,2,3"
```

### Config

This repository is used `mmdetection, mmsegmentation ...` like config managements.

```yaml
__BASE__:
  - config/__BASE__/config.yaml
  - config/__BASE__/OPTIMIZER/Momentum.yaml
  - config/__BASE__/LR_SCHEDULER/ReduceLROnPlateau.yaml

__DATASET__: config/__BASE__/DATASET/dataset.yaml

EPOCH: 12

# MODEL Param
MODEL:
  NAME: Model_Name
  MODEL: Model # Class名
  INPUT_SIZE: [224, 224]
  PRE_TRAINED: True
  PRE_TRAINED_WEIGHT: result/PreTrain/efficientdet-d0.pth
  # Load Weight Path
  WEIGHT: "./"
  SYNC_BN: False
  TIMM_SYNC_BN: False
  FIND_UNUSED_PARAMETERS: False
  LOSS: loss
```

## Structure

```bash
.
├── config/ # yaml like config file(OmegaConf)
├── MODEL
│   └── model.yaml
└── __BASE__
    ├── DATASET
    ├── LR_SCHEDULER
    ├── OPTIMIZER
    └── config.yaml
├── dataset -> ../dataset
├── doc/
├── docker/
│   ├── Dockerfile
│   ├── build.sh
│   └── run.sh
├── etc/
├── notebook/
├── script/
├── src # src code for not execute
│   ├── dataloaders/
│   ├── losses/
│   ├── models/
│   ├── optimizer/
│   ├── transform/
│   ├── utils/
│   ├── sampler.py
│   ├── scheduler.py
│   ├── tester.py # Test class
│   └── trainer.py # Training class
├── tests # script for test code
│   └── dev # script for development
├── README.md
├── clean_result.py
├── mlflow_parse.py
├── mlflow_to_csv.py
├── poetry.lock
├── pyproject.toml
├── template.env
├── test.py
├── torchrun.sh
└── train.py
```
