# quick_demo.py
# 目的：製造部品や車の画像をAIで分類する基本的な仕組みを理解するデモコード
# 最小構成（畳み込み→活性化→プーリング→全結合）
# 3エポックだけの軽学習（1分以内に終わる目安）

# 1. 必要なライブラリの読み込み
import torch       # AI計算ライブラリ
import torch.nn as nn # ニューラルネットワーク構成用
import torch.optim as optim # 最適化（学習を進めるためのアルゴリズム）
from torchvision import datasets, transforms # 画像データの読み込み＆前処理
from torch.utils.data import DataLoader # データをまとめて扱うクラス

# 2. データの前処理を設定
transform = transforms.Compose([
    transforms.Resize((64, 64)), # 画像を64×64にそろえる
    transforms.ToTensor(),       # 画像を数値データに変換
])

# 3. データを読み込む（ここでは例としてフォルダ構造データを想定）
#    data/train/とdata/test/の中にクラスごとフォルダを置けばOKです
train_data = datasets.ImageFolder(R"C:\work\mazda_ai_portfolio\datasets\mvtec_ad\processed\metal_nut\metal_nut\train", transform=transform)
test_data = datasets.ImageFolder(R"C:\work\mazda_ai_portfolio\datasets\mvtec_ad\processed\metal_nut\metal_nut\train", transform=transform)

# 4. データをバッチごとにまとめて扱うDataLoaderを作成
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# 5. シンプルなCNNモデルを定義（画像を2分類する想定）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 画像を見る「目」の部分：畳み込み層conv2つ
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), # RGB画像→8枚の特徴マップへ
            nn.ReLU(),                     # 活性化関数で非線形に変換
            nn.MaxPool2d(2, 2),            # 画像を半分に縮小
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),                     # 活性化関数で非線形に変換
            nn.MaxPool2d(2, 2), 
        )
        # 最後の判断部分：全結合層fc
        self.fc = nn.Sequential(
            nn.Flatten(),           # 平らにする
            nn.Linear(16 * 16 * 16, 32), # 中間層
            nn.ReLU(),
            nn.Linear(32, 2)          # 出力(2クラス：良品・不良品)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPUが使用できる場合はGPUへ転送
model = SimpleCNN().to(device)                                        # モデルのインスタンス化

# 6. 学習設定
loss_fn = nn.CrossEntropyLoss()                # 損失関数で、予測と正解の違いを数値化
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 学習ループ
for epoch in range(3): # 3エポック繰り返し
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        preds = model(images)           # モデルに画像を入力して、予測を出力
        loss = loss_fn(preds, labels)   # 損失を計算
        optimizer.zero_grad()           # 勾配をリセット
        loss.backward()                 # 誤差を逆伝搬
        optimizer.step()                # 重みを更新
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

# 8. 評価処理
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        predicted = torch.argmax(preds, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"正解率: {100 * correct / total:.1f}%")

# 9. モデル保存
torch.save(model.state_dict(), "../results/simplecnn_demo.pth")
print("モデルを保存しました！")
print("success")