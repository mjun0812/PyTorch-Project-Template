# PyTorch Project Template

PyTorchのProjectテンプレートです．

## Features

- Docker + uvで環境構築
- PyTorchのDistributed Data Parallel(DDP), Data Parallel, Fully Shared Distributed Parallel(FSDP)によるマルチGPU Training
- [MLflow](https://mlflow.org)と[wandb](https://www.wandb.jp)を使った実験管理
- [OmegaConf](https://github.com/omry/omegaconf)を使ったコンフィグ管理

## Environments

- Python 3.11
- CUDA 12.4
- PyTorch 2.5.0

## Install

環境構築はDockerで行います．
Dockerコンテナに`./dataset`をマウントするため，この中に各データセットのディレクトリを入れてください。
Docker Imageのビルドは以下のコマンドで行えます。

```bash
./docker/build.sh
```

### MLflow

本テンプレートでは，MLflowによる実験管理が行えます。
MLflowのデータをローカルに保存する場合と，外部のサーバに送信する場合の両方に対応しています．
デフォルトはローカル保存となり，`result/mlruns`に保存されます．

外部サーバを利用する場合は，dotenvに設定を書き込む必要があります．
`template.env`をコピーして利用してください．

```bash
$ cp template.env .env
$ vim .env

SLACK_TOKEN="HOGE"
MLFLOW_TRACKING_URI=""

# Basic Auth
# MLFLOW_TRACKING_USERNAME=""
# MLFLOW_TRACKING_PASSWORD=""
```

ローカルで保存している場合，mlflow server(ui)のコマンドは以下です．

```bash
./docker/run.sh --mlflow-ui ./script/run_mlflow.sh
```

### Optional: Wandb

Wandbによる実験管理も行えます。
`.env`の`WANDB_API_KEY`を設定し、configの`wandb.use`をtrueにすれば結果が送信されます。

```bash
./docker/run.sh python train.py config/dummy.yaml wandb.use=true wandb.project_name="hoge"
```

### Optional: Slackによる通知

学習の終了や評価の終了時にSlackに通知を行うことができます．
通知を行うには.envにSlackのトークン(Webhookではない)を書き込む必要があります．
デフォルトでは，通知は`channel="#通知", username="通知"`で行われます．
`.env`の`SLACK_TOKEN`にAPI tokenを入れて下さい。

## Usage

Dockerコンテナ内でコマンドを実行します．
そのため，実行するコマンドの先頭に`./docker/run.sh`をつけてください．

```bash
./docker/run.sh python train.py config/model/model.yaml gpu.use=1
./docker/run.sh python test.py result/20220911/config.yaml gpu.use=1
```

yaml内のConfigの値をCLIから変更することもできます．以下のように，`.`で連結して，`=`で値を指定してください．

```yaml
gpu:
  use: 1
```

```bash
./docker/run.sh python train.py config/model/ResNet.yaml gpu.use=2
```

### Train

学習を行う場合は，`train.py`を使います．このスクリプトでは，学習が終わった後にテストも一緒に行われます．
`train.py`では`config`以下のyamlファイルを指定します．

```bash
./docker/run.sh python train.py config/model/ResNet.yaml
```

複数GPUを用いた学習を行う場合は，実行するコマンドの`python`を消して，
前に`./torchrun.sh [GPU数]`を入れ，`gpu.use="0,1"`のように，Configの値を変更します．
この時，GPUのIDの順番は`nvidia-smi`コマンドで並ぶPCIeの順番になっています．

```bash
./docker/run.sh ./torchrun.sh 4 train.py config/model/ResNet.yaml gpu.use="0,1,2,3"
```

学習結果は`result/[train_dataset.name]/[日付]_[model.name]_[dataset.name]_[tag]`以下のディレクトリに保存されます．

### Test

評価を行うスクリプトは`test.py`です．
学習で動かす`train.py`でも評価は実行されますが，手動で行う場合はこちらを使用してください．
`test.py`では，学習結果の保存されているディレクトリにある`config.yaml`を第1引数に指定します．

```bash
./docker/run.sh python test.py result/ImageNet/hoge_hoge/config.yaml gpu.use=7
```

上記のコマンドは，学習時のログでも表示されています．学習時のログは，
`result/[train_dataset.name]/[日付]_[model.name]_[dataset.name]_[tag]/train.log`に保存されています．

テスト時のログやデータは
`result/[train_dataset/name]/[日付]_[model.name]_[dataset.name]_[tag]/runs/`以下に，
ディレクトリが作成され，その中に保存されます．

## Scripts

### Config一括編集

```bash
./docker/run.sh python script/edit_configs.py [config_path or recursive directory] "params.hoge=aa,params.fuga=bb"
```

### MLflow UI

```bash
./docker/run.sh --mlflow-ui ./script/run_mlflow.sh
```

### JupyterLab

```bash
./script/run_notebook.sh
```

### Test

```bash
./docker/run.sh ./script/run_test.sh
```

### Clean result

MLflowで削除された実験結果をローカルの`./result/`から削除する

```bash
./docker/run.sh
# In container
python script/clean_result.py | xargs -I{} -P 2 rm -rf {}
```

### 集計

MLflowの結果を集計し、`./doc/result_csv`以下に保存する。

```bash
./docker/run.sh python script/aggregate_mlflow.py [dataset_name or all]
```
