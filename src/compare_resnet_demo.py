# compare_resnet_demo.py
# 目的：自作CNNとResNet18を使って画像分類を比較するデモコード
"""
自作CNNは「構造の理解」に重きを置いたシンプルな層構成。

ResNet18は「精度の高い実用モデル」として比較対象に。

学習エポックを2回に短縮して、1回数分で確認可能。

精度（Accuracy）も即時表示されるため、学習効果がすぐ見える。
"""

# 1. 必要なライブラリをインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 2. 画像前処理（同じ処理をモデルにも使う）
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 画像を64x64に
    transforms.ToTensor(),        # 数値データに変換
])

# 3. データ読み込み（data/train, data/testを使用）
train_data = datasets.ImageFolder(R"C:\work\mazda_ai_portfolio\datasets\mvtec_ad\processed\metal_nut\metal_nut\train", transform=transform)
test_data = datasets.ImageFolder(R"C:\work\mazda_ai_portfolio\datasets\mvtec_ad\processed\metal_nut\metal_nut\val", transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# 4. 自作CNNモデル
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 5, ResNet18モデルを準備（学習済みモデルを利用）
def create_resnet(num_classes):
    model = models.resnet18(pretrained=True)    # 事前学習済みモデル
    model.fc = nn.Linear(model.fc.in_features, num_classes) # 出力層の差し替え
    return model

# 6. モデルの作成
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simple_model = SimpleCNN(num_classes=2).to(device)
resnet_model = create_resnet(num_classes=2).to(device)

# 7. 学習用関数定義
def train(model, loader, name):
    print(f"{name} の学習を開始...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):  # 少し短くする
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{name}] Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f}")

# 8. 簡単な評価関数
def evaluate(model, loader, name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"[{name}] 正答率: {acc:.1f}%")

# 9. 実際に学習・評価を実行
train(simple_model, train_loader, "自作CNN")
evaluate(simple_model, test_loader, "自作CNN")

train(resnet_model, train_loader, "ResNet18")
evaluate(resnet_model, test_loader, "ResNet18")

# 10. モデル保存
torch.save(simple_model.state_dict(), "../results/simplecnn_vs_resnet_simplecnn.pth")
torch.save(resnet_model.state_dict(), "../results/simplecnn_vs_resnet_resnet.pth")
print("両方のモデルを保存しました。")
print("ok")