# CNNモデル定義
# 必要なライブラリをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

# シンプルな自作CNNモデルを定義
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        # nn.Moduleクラスの継承を初期化
        super(SimpleCNN, self).__init__()
        # 1層目：入力画像(3ch)、出力16chの畳み込み層。カーネルサイズ3x3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 2層目：入力16ch、出力32ch
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # MaxPooling層（2x2でダウンサンプリング＝画像を1/2サイズに）
        self.pool = nn.MaxPool2d(2, 2)
        # 全結合層（特徴をクラスに変換）
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 入力次元は画像サイズ次第で調整必要
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # ReLUで活性化し、プーリングでサイズを縮小
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 特徴マップを平坦化
        x = x.view(-1, 32 * 8 * 8)
        # 全結合層を通して分類
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ResNetベースのモデルを作成する関数
def build_resnet_model(num_classes):
    # ImageNetで学習済み重みを使う
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # 最終層（出力層）を自分のクラス数に置き換える
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

