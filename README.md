# PyTorch Project Template

PyTorchのProjectテンプレートです．

## Features

- Rye + Dockerで環境構築
- PyTorchのDistributed Data Parallel(DDP)とData ParallelによるマルチGPU Training
- `torch.amp`を使った混合精度学習(FP16, FP32)
- `torch.compile`に対応
- [MLflow](https://mlflow.org)を使った実験管理
- [OmegaConf](https://github.com/omry/omegaconf)を使ったmmcv Likeなyamlコンフィグ管理

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

デバッグ等でMLflowによる実験管理をオフにしたい場合は，`USE_MLFLOW=false`を実行コマンドに追加してください．

```bash
./docker/run.sh python train.py config.yaml USE_MLFLOW=False
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
./docker/run.sh python train.py config/MODEL/model.yaml GPU.USE=1
./docker/run.sh python test.py result/20220911/config.yaml GPU.USE=1
```

yaml内のConfigの値をCLIから変更することもできます．以下のように，`.`で連結して，`=`で値を指定してください．

```yaml
GPU:
  USE: 1
```

```bash
./docker/run.sh python train.py config/MODEL/ResNet.yaml GPU.USE=2
```

### Train

学習を行う場合は，`train.py`を使います．このスクリプトでは，学習が終わった後にテストも一緒に行われます．
`train.py`では`config`以下のyamlファイルを指定します．

```bash
./docker/run.sh python train.py config/MODEL/ResNet.yaml
```

複数GPUを用いた学習を行う場合は，実行するコマンドの`python`を消して，
前に`./torchrun.sh [GPU数]`を入れ，`GPU.USE="0,1"`のように，Configの値を変更します．
この時，GPUのIDの順番は`nvidia-smi`コマンドで並ぶPCIeの順番になっています．

```bash
./docker/run.sh ./torchrun.sh 4 train.py config/MODEL/ResNet.yaml GPU.USE="0,1,2,3"
```

学習結果は`result/[Dataset名]/[日付]_[モデル]_[データセット]_[タグ]`以下のディレクトリに保存されます．

### Test

評価を行うスクリプトは`test.py`です．
学習で動かす`train.py`でも評価は実行されますが，手動で行う場合はこちらを使用してください．  
`test.py`では，学習結果の保存されているディレクトリにある`config.yaml`を第1引数に指定します．

```bash
./docker/run.sh python test.py result/ImageNet/hoge_hoge/config.yaml GPU.USE=7
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

## Config

本リポジトリでは，Configの管理をOmegaConfライブラリを用いたyamlファイルで管理しています．  
MMLab系のフレームワークのように，他のyamlファイルのConfigの継承とimportを採用しています．

例として，以下の3つのyamlがあるとします．

```yaml
# config/__BASE__/DATASET/dataset.yaml
NAME: dataset_name
ROOT: ./dataset/ImageNet/

TRANSFORMS:
  TRAIN:
    - name: Resizer
      args:
        img_scale: [224, 224]
    - name: NormalizeImage
      args: null
  VAL:
    - name: Resizer
      args:
        img_scale: [224, 224]
    - name: NormalizeImage
      args: null
  TEST:
    - name: Resizer
      args:
        img_scale: [224, 224]
    - name: NormalizeImage
      args: null
```

```yaml
# config/__BASE__/OPTIMIZER/momentum.yaml
NAME: Momentum
LR: 1e-3
MOMENTUM: 0.937
WEIGHT_DECAY: 4e-5
```

```yaml
# config/MODEL/model.yaml
__BASE__:
  - config/__BASE__/OPTIMIZER/Momentum.yaml

__TRAIN_DATASET__: config/__BASE__/DATASET/dataset.yaml
__VAL_DATASET__: config/__BASE__/DATASET/dataset.yaml
__TEST_DATASET__: config/__BASE__/DATASET/dataset.yaml

EPOCH: 12

OPTIMIZER:
  LR: 5e-5
```

`train.py`で第1引数に指定するのは，`python train.py config/MDOEL/model.yaml`です．  
`__BASE__`に指定したyamlの値が追加されます．
追加される時には，`__BASE__`の1つ下のディレクトリ名がKeyとなります．
上記の例だと，`config/__BASE__/OPTIMIZER/Momentum.yaml`の中の値は，
`OPTIMIZER:`以下に設定されます．

また，`config/MODEL/model.yaml`で設定した値が最優先されます．上記の例では，`OPTIMIZER.LR`を書き換えているので，下記の結果では値が上書きされています．

```yaml
EPOCH: 12

OPTIMIZER:
  NAME: Momentum
  LR: 5e-5
  MOMENTUM: 0.937
  WEIGHT_DECAY: 4e-5

TRAIN_DATASET:
  NAME: dataset_name
  ROOT: ./dataset/ImageNet/

  TRANSFORMS:
    TRAIN:
      - name: Resizer
        args:
          img_scale: [224, 224]
      - name: NormalizeImage
        args: null
    VAL:
      - name: Resizer
        args:
          img_scale: [224, 224]
      - name: NormalizeImage
        args: null
    TEST:
      - name: Resizer
        args:
          img_scale: [224, 224]
      - name: NormalizeImage
        args: null
```

## よく使うConfigの値

| 値                  | 説明                                                                         | 例                                                                                         |
|---------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `GPU.USE`           | 使用するGPUのIDを指定します                                                         | GPU.USE="[0,1]", GPU.USE=1                                                                 |
| `BATCH`             | バッチ数を設定します                                                                | ./docker/run.sh python train.py BATCH=2                                                    |
| `EPOCH`             | エポック数を指定します                                                               | ./docker/run.sh python train.py EPOCH=100                                                  |
| `__TRAIN_DATASET__` | 学習するデータセットを変更します．一緒に`__VAL_DATASET__`も変更してください．値はファイルパスを入力してください． | `./docker/run.sh python train.py __TRAIN_DATASET__="config/__BASE__/DATASET/dataset.yaml"` |
| `OPTIMIZER.LR`      | 学習率を変えます                                                                 | OPTIMIZER.LR=2e-4                                                                          |
| `MODEL.WEIGHT`      | テストに使う重みファイルを指定します．`test.py`と一緒に使用して下さい                             | MODEL.WEIGHT=./model.pth                                                                   |

## 便利スクリプト

デバッグや確認がしやすいスクリプトを用意しています．

- `mlflow_parse.py`

mlflowの結果をデータセットでフィルタして表で表示できます．  
`python mlflow_parse.py "Dataset Name or all"`というように第1引数にデータセットの名前か`all`を入れて使います．  
表示する項目はタスクによって異なると思うので，このスクリプトを編集して調整してください．

- `clean_result.py`

mlflowのWebUIで削除した実験を，`result`以下からも削除するためのスクリプトです．  
このスクリプトではディレクトリのパスを表示するだけなので，`xargs`と組み合わせて削除してください．

## テスト

`tests`ディレクトリ以下は，実装が正しく行われているかをチェックするために使用します．

- `test_config.py`

Yamlファイルを引数として，実際にどのようなConfigが展開されるかを確認することができます．

```bash
./docker/run.sh python tests/test_config.py config/MODEL/model.yaml
```

- `test_model.py`

`config/MODEL`以下のyamlファイルを第1引数として，実際にモデルの呼び出し，データセットの呼び出し，損失関数の呼び出しを行い，それぞれが動作するかを確認します．

```bash
./docker/run.sh python tests/test_model.py config/MODEL/model.yaml
```

- `test_all_model.py`

`config/MODEL/`以下の全てのyamlファイルを対象として，`test_model.py`を実行します．
実行結果は以下のようになります．

```bash
python tests/test_all_model.py

./docker/run.sh python tests/test_model.py config/MODEL/BiFPNModel/BiFPNModel.yaml: Passed
Failed: ./docker/run.sh python tests/test_model.py config/MODEL/CascadeRCNN/CascadeRCNN_swin.yaml
```

pythonから各モデルに対してスクリプトが実行されて，実行が成功するかを検証します．

- `test_dataloader.py`

dataloaderをイテレーションして，どのくらいの時間がかかるかを検証します．

```bash
./docker/run.sh python tests/test_dataloader.py config/MODEL/model.yaml
```

- `test_lr_scheduler.py`

`config/__BASE__/LR_SCHEDULER`以下で定義されているLR Schedulerの挙動をグラフで確認します．  
結果は，`doc/lr_scheduler`以下に保存されます．

- `test_transform.py`

画像に対して，実際にどのようなAugmentationが行われるかを確認することができます．
学習，評価時の切り替えを行うには，`PHASE=val`と設定します．

```bash
./docker/run.sh python tests/test_transform.py config/MODEL/model.yaml PHASE=val
```
