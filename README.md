# PyTorch Project Template

PyTorchのProjectテンプレートです．

## Features

- Rye + Dockerで環境構築
- PyTorchのDistributed Data Parallel(DDP)とData ParallelによるマルチGPU Training
- `torch.amp`を使った混合精度学習(FP16, FP32)
- `torch.compile`に対応
- [MLflow](https://mlflow.org)を使った実験管理
- [OmegaConf](https://github.com/omry/omegaconf)を使ったコンフィグ管理

## Environments

- Python 3.11
- CUDA 12.1
- PyTorch
- TorchVision

## Install

環境構築はDockerで行います．
Dockerコンテナにデータセットディレクトリをマウントするため，先にディレクトリを作成するか，シンボリックリンクを作成しておきます．

```bash
ln -sfv [datasets_dir] ./dataset
```

次に，Dockerイメージをビルドします．ビルドスクリプトが用意されています．

```bash
./docker/build.sh
```

### Optional: MLflow

本テンプレートでは，MLflowによる実験管理を行います．
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
Dockerでポートフォワードを行っていないので，ホストマシンでmlflowを実行する必要があります．

```bash
cd result
mlflow ui --port 5000
```

### Optional: Slackによる通知

学習の終了や評価の終了時にSlackに通知を行うことができます．
通知を行うには.envにSlackのトークン(Webhookではない)を書き込む必要があります．
デフォルトでは，通知は`channel="#通知", username="通知"`で行われます．
`.env`に以下の値を記入してください．

```bash
SLACK_TOKEN="HOGE"
```

## Use as Template

本テンプレートを用いてリポジトリを作成するときは，本リポジトリをcloneして，
リモートリポジトリを変更して利用してください．

```bash
# Clone
git clone git@github.com:mjun0812/PyTorch-Project-Template.git [project_name]
cd [project_name]

# デフォルト(origin)のリモートリポジトリをupstreamに名前変更
git remote rename origin upstream

# 自分のリモートリポジトリを追加
git remote add origin [uri]
```

テンプレートは度々更新されます．その時，プロジェクトに更新を反映するには，
fetchとmergeで反映します．
自分で一部のコードを変更していた場合は，コンフリクトが発生するかもしれないです．

```bash
git fetch upstream main
git merge upstream/main
```

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

学習結果は`result/[Dataset名]/[日付]_[モデル]_[データセット]_[タグ]`以下のディレクトリに保存されます．

### Test

評価を行うスクリプトは`test.py`です．
学習で動かす`train.py`でも評価は実行されますが，手動で行う場合はこちらを使用してください．
`test.py`では，学習結果の保存されているディレクトリにある`config.yaml`を第1引数に指定します．

```bash
./docker/run.sh python test.py result/ImageNet/hoge_hoge/config.yaml gpu.use=7
```

上記のコマンドは，学習時のログでも表示されています．学習時のログは，
`result/[Dataset名]/[日付]_[モデル]_[データセット]_[タグ]/train.log`に保存されています．

```bash
[2023-11-30 13:06:26,337][INFO] Finish Training {'Test Cmd': 'python test.py result/hoge/fuga/config.yaml',
 'Train save': 'result/hoge/fuga',
 'Val Loss': '  9.013',
 'dataset': 'hoge',
 'host': 'server',
 'model': 'fuga',
 'tag': ''}
```

テスト時のログやデータは`result/[Dataset名]/[日付]_[モデル]_[データセット]_[タグ]/runs/`以下に，
`[日付]_[使用した重みの名前]`のようなディレクトリが作成され，その中に保存されます．

