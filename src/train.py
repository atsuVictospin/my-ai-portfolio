# 学習ループの実装
# dataset.pyとmodel.pyを呼ぼだし、学習設定から実行までを担う。
import torch
from torch import nn, optim
from tochvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN,build_resnet_model

# トレーニング処理関数
def tarin_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()    # 勾配を初期化
            outputs = model(images)  # 画像をモデルに入力
            loss = criterion(outputs, labels) # 損失を計算
            loss.backward()                   # 誤差逆伝搬
            optimizer.step()                  # パラメータ更新
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    print("Training finished!")