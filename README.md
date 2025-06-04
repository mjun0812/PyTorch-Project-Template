# PyTorch Project Template

PyTorchのProjectテンプレートです．

## Features

- Docker + uvで環境構築
- Multi Node, Multi GPU Trainingのサポート
- PyTorchのDistributed Data Parallel(DDP), Data Parallel, Fully Shared Distributed Parallel(FSDP)によるマルチGPU Training
- devcontainer
- [MLflow](https://mlflow.org)と[wandb](https://www.wandb.jp)を使った実験管理
- [OmegaConf](https://github.com/omry/omegaconf)を使ったコンフィグ管理
- データセットの一部をRAMにキャッシュする機能
- 学習の再開機能

## Environments

- Python 3.11
- CUDA 12.8
- PyTorch 2.7.0

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
./docker/run.sh python train.py [config_file_path]
./docker/run.sh python train.py config/model/ResNet.yaml
```

学習結果は`result/[train_dataset.name]/[日付]_[model.name]_[dataset.name]_[tag]`以下のディレクトリに保存されます．

#### Single Node Multi GPU Training

シングルノード・複数GPUを用いた学習を行う場合は，
実行するコマンドの`python`を消して，
前に`./torchrun.sh [GPU数]`を入れ，`gpu.use="0,1"`のように，Configの値を変更します．
この時，GPUのIDの順番は`nvidia-smi`コマンドで並ぶPCIeの順番になっています．

```bash
./docker/run.sh ./torchrun.sh 4 train.py config/model/ResNet.yaml gpu.use="0,1,2,3"
```

#### Multi Node Multi GPU Training

マルチノード・複数GPUを用いた学習を行う場合は，
実行するコマンドの`python`を消して，
前に`./multinode.sh [ノード数] [GPU数] [ジョブID] [ノードランク] [マスターノードのホスト名:マスターノードのポート]`を入れ，
`gpu.use="0,1"`のように，Configの値を変更します．

```bash
# Master Node
./docker/run.sh ./multinode.sh 2 4 12345 0 localhost:12345 train.py config/model/ResNet.yaml gpu.use=0,1,2,3

# Worker Node
./docker/run.sh ./multinode.sh 2 4 12345 1 192.168.1.10:12345 train.py config/model/ResNet.yaml gpu.use=4,5,6,7
```

#### Train Option: RAM Cache

データセットの一部をRAMにキャッシュする機能があります。キャッシュは`torch.Tensor`のみ対応しています。

```bash
./docker/run.sh python train.py config/model/ResNet.yaml gpu.use=1 use_ram_cache=true ram_cache_size_gb=16
```

この機能を使うには、datasetの実装を以下のように工夫する必要があります。

```python
if self.cache is not None and idx in self.cache:
    image = self.cache.get(idx)
else:
    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    if self.cache is not None:
        self.cache.set(idx, image)
```

#### Train Option: Resume Training

学習の終了後、もしくは学習を中断した場合に、
結果のディレクトリに保存されている`config.yaml`である、
`result/[train_dataset.name]/[日付]_[model.name]_[dataset.name]_[tag]/config.yaml`を指定して実行すると，学習を再開できます。
この時、元のepochが100だった場合でも、`epoch=150`をコマンドラインで指定して実行すると、configが上書きされて150epochまで学習が継続されます。

```bash
./docker/run.sh python train.py config/config.yaml # 100epochまで完了

# 上記の結果を利用して学習を再開 or 継続
./docker/run.sh python train.py result/ImageNet/hoge_hoge/config.yaml epoch=150 gpu.use=7 # 150epochまで学習を継続
```

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

### Test Source

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

### 実装済みモジュールの確認

```bash
python script/show_options.py

DATASET_REGISTRY
Registry of DATASET:
╒══════════════╤════════════════════════════════════════════════╕
│ Names        │ Objects                                        │
╞══════════════╪════════════════════════════════════════════════╡
│ DummyDataset │ <class 'src.dataloaders.dataset.DummyDataset'> │
╘══════════════╧════════════════════════════════════════════════╛
EVALUATOR_REGISTRY
Registry of EVALUATOR:
╒════════════════╤══════════════════════════════════════════════════╕
│ Names          │ Objects                                          │
╞════════════════╪══════════════════════════════════════════════════╡
│ DummyEvaluator │ <class 'src.evaluator.evaluator.DummyEvaluator'> │
╘════════════════╧══════════════════════════════════════════════════╛
MODEL_REGISTRY
Registry of MODEL:
╒════════════╤═══════════════════════════════════════╕
│ Names      │ Objects                               │
╞════════════╪═══════════════════════════════════════╡
│ DummyModel │ <class 'src.models.model.DummyModel'> │
╘════════════╧═══════════════════════════════════════╛
BACKBONE_REGISTRY
['bat_resnext26ts',
 'beit_base_patch16_224',
 'beit_base_patch16_384',
 'beit_large_patch16_224',
...
```

## Structure

```bash
.//
├── config/            # 実験とモデルの設定ファイル
│   └── __base__/      # 基本設定ファイル
├── dataset/           # データセットを保存するディレクトリ（Dockerコンテナにマウント）
├── doc/               # ドキュメントファイルと実験結果
├── docker/            # Dockerセットアップとユーティリティスクリプト
├── etc/               # その他のファイル
├── notebook/          # 探索と分析用のJupyterノートブック
├── result/            # 学習結果、チェックポイント、ログ
├── script/            # 様々なタスク用のユーティリティスクリプト
│   ├── aggregate_mlflow.py  # MLflow結果を集計するスクリプト
│   ├── clean_result.py      # resultディレクトリをクリーンアップ
│   ├── edit_configs.py      # 設定ファイルの一括編集
│   ├── show_*.py            # 様々な情報を表示するスクリプト
│   ├── run_mlflow.sh        # MLflow UIを起動
│   ├── run_notebook.sh      # JupyterLabを起動
│   └── run_test.sh          # テストを実行
├── src/               # ソースコード
│   ├── config/        # 設定の処理
│   ├── dataloaders/   # データ読み込みユーティリティ
│   ├── evaluator/     # モデル評価コード
│   ├── models/        # モデル定義
│   ├── optimizer/     # 最適化アルゴリズム
│   ├── scheduler/     # 学習率スケジューラ
│   ├── transform/     # データ変換
│   ├── utils/         # ユーティリティ関数
│   ├── sampler.py     # データサンプリングユーティリティ
│   ├── tester.py      # テストループの実装
│   ├── trainer.py     # 学習ループの実装
│   └── types.py       # 型定義
├── tests/             # テストコード
├── README.md
├── template.env       # 環境変数のテンプレート
├── train.py
├── test.py
├── torchrun.sh        # 単一ノードでの分散学習用スクリプト
├── multinode.sh       # マルチノード分散学習用スクリプト
├── pyproject.toml
└── uv.lock
```
