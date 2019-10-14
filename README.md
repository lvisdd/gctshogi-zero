# gctshogi-zero

[wcsc29アピール文書](https://gist.github.com/lvisdd/9b49ab88600fa242f2138fad4eb06caf)

## 使用ライブラリ
* [python-shogi](https://github.com/gunyarakun/python-shogi)
* [python-dlshogi](https://github.com/TadaoYamaoka/python-dlshogi)

## ビルド環境
### USIエンジン、自己対局プログラム
#### Windowsの場合
* Windows 10 64bit
* Visual Studio 2015
#### Linuxの場合 (Google Colab)
* Ubuntu 18.04 LTS
#### Windows、Linux共通
* CUDA 10.0
* cuDNN 7.5

### 学習部
上記USIエンジンのビルド環境に加えて以下が必要
* Python 3.6 ([Anaconda](https://www.continuum.io/downloads) 4.5.2 (64-bit))
* tensorflow-gpu
* keras
* cython

## ライセンス
ライセンスはMITライセンスとします。

# Windows 環境構築

## Anaconda 環境構築

``` dos
## For Python 3.6
conda create --name=gctshogi-zero python=3.6
activate gctshogi-zero
```

## GitHub リポジトリのクローン

``` dos
mkdir c:\work
cd c:\work
git clone https://github.com/lvisdd/gctshogi-zero.git
cd gctshogi-zero
```

## パッケージのインストール

``` dos
pip install tensorflow-gpu
pip install keras
pip install matplotlib
pip install cython

#pip install python-shogi
pip install git+https://github.com/lvisdd/python-shogi.git --no-cache-dir

pip install --no-cache-dir -e .
```

## 事前準備

* [floodgate] (https://ja.osdn.net/projects/shogi-server/releases/) の棋譜をダウンロード
* [7-Zip] (https://sevenzip.osdn.jp/) で解凍する

``` dos
mkdir data\floodgate
cd data\floodgate
wget http://iij.dl.osdn.jp/shogi-server/68500/wdoor2016.7z
7z x wdoor2016.7z

-> C:\work\gctshogi-zero\data\floodgate\2016
```

* 棋譜リストのフィルター／ファイル準備

``` dos
cd C:/work/gctshogi-zero
python ./utils/filter_csa_in_dir.py C:/work/gctshogi-zero/data/floodgate/2016
python ./utils/prepare_kifu_list.py C:/work/gctshogi-zero/data/floodgate/2016 kifulist
```

## 学習

``` dos
### GPU
python ./train.py kifulist_train.txt kifulist_test.txt ./model/model-2016.h5 --batchsize 256

### TPU
python ./train.py kifulist_train.txt kifulist_test.txt ./model/model-2016.h5 --batchsize 1024 --use_tpu
```

## 学習の継続

``` dos
### GPU
python ./train_from_csa.py kifulist_train.txt kifulist_test.txt ./model/model-2017.h5 --batchsize 256 --resume ./model/model-2016.h5

### TPU
python ./train_from_csa.py kifulist_train.txt kifulist_test.txt ./model/model-2017.h5 --batchsize 1024 --use_tpu --resume ./model/model-2016.h5
```

## Kerasモデル -> Tensorflowモデルを保存する場合

```
python convert_model_k2tf.py -r ./model/model-2016.h5 model

-> model\1556503437
   * assets
   * saved_model.pb
   * variables
```

## 将棋所で対局

* 将棋所を[ダウンロード] (http://shogidokoro.starfree.jp/download.html) して任意のディレクトリに解凍
* Shogidokoro.exe を起動
* 「対局」メニューの「エンジン管理」を選択
* 「C:\work\DeepLearningShogi\bat\parallel_mcts_player.bat」を「追加」
* 「閉じる」を選択
* 「対局」メニューの「対局」を選択
* 先手または後手の「エンジン」－「parallel_mcts_player」を選択し、「OK」を選択

