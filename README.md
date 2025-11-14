# 製造現場画像分類AIプロジェクト（マツダAIシステム部応募用）

## プロジェクト概要
本プロジェクトは、製造現場での部品が造を分類する簡易AIモデルを構築し、品質管理の省力化と業務効率向上を目指します。
Pytorchライブラリを用いて、シンプルなCNNを実装し、画像分類を実現しました。


## 実装環境
- python=3.9.25
- PyTorch 2.8.0 + cu128
↓インストールした際のコード
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
(GPU利用可)

## データセット
- 使用データ：MVTec AD公開製造業画像データセット  
- クラス数：良品、不良品の計2クラス  
- データ構成：次のフォルダごとに画像を配置
            `dataset/raw/train/`, `dataset/raw/val/`  
- ラベル管理：`dataset/labels.csv`に、ファイル名とラベルの対応を保存


## 環境構築
```
1.Anaconda Prompt またはターミナル以下を入力
conda env create -f environment.yml
または手動で
pip install -r requirements.txt

2.Jupyter Notebook 開始（ymlファイルを読み込むと環境名が ai-portfolio になる）
conda activate "仮想環境名"
jupyter notebook
```

## 実行方法
### 1.学習させる
```
python src/train.py --epochs 20 --batch_size 32 --lr 0.001
```
### 2.推論させる(例：sample.jpg)
```
python src/inference.py --image data/test/sample.jpg
```

## モデル構成
- 基底モデル：ResNet18（事前学習重み利用）
- 出力層：2クラス分類用に変更

## 結果
- 精度（Accuracy）：約〇〇%  
- F1スコア：0.00  
- 混同行列やROC曲線は`results/`に保存

## 今後の課題
- 様々なデータセットへの汎用性向上  
- モデル軽量化
- 推論高速化  


## 作者
- 氏名：阪本 惇暉  
- 連絡先：atsuprogramming119@gmail.com

## ライセンス
MIT License

## 参考文献
- https://qiita.com/mathlive/items/8e1f9a8467fff8dfd03c
- https://techgym.jp/column/pytorch-cnn/
- https://ex-ture.com/blog/2021/01/11/pytorch-cnn/
