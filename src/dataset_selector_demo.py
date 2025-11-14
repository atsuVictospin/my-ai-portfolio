# dataset_selector_demo.py
# 目的：MVTec AD（製造部品の良品・不良品分類）とStanford Cars（車種分類）を
#       同じAIモデル構造で切り替えて学習できるテンプレート
"""
dataset_type = "mvtec" → 製造部品の良不良分類
dataset_type = "cars" → 車種分類

それぞれのフォルダ構造は以下のように配置。
data/raw/mvtec_ad/train/
    ├── good/
    └── defect/
data/raw/mvtec_ad/test/
    ├── good/
    └── defect/
data/raw/stanford_cars/train/
    ├── Acura_NSX/
    ├── BMW_3_Series/
    └── ...
data/raw/stanford_cars/test/
    ├── Acura_NSX/
    ├── BMW_3_Series/
    └── ...

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

# ------------------------------
# ① どちらのデータを使うか選択する
# ------------------------------
# "mvtec" にすると製造部品データ、
# "cars" にするとStanford Carsデータで分類
dataset_type = "mvtec"  # ← "cars" に変えると車種分類になる

# ------------------------------
# ② データパスを自動で切り替え
# ------------------------------
if dataset_type == "mvtec":
    train_path = R"datasets\mvtec_ad\processed\metal_nut\metal_nut\train"
    test_path = R"datasets\mvtec_ad\processed\metal_nut\metal_nut\val"
    num_classes = 2  # 良品 or 不良品
    print("MVTec AD データセットを使用します。")
else:
    train_path = "data/raw/stanford_cars/train"
    test_path = "data/raw/stanford_cars/test"
    num_classes = 196  # Stanford Cars のクラス数（196種）
    print("Stanford Cars データセットを使用します。")

# ------------------------------
# ③ 前処理設定（画像サイズ調整など）
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 画像を128×128に統一
    transforms.ToTensor(),          # 数値に変換
])

# ------------------------------
# ④ データ読み込み
# ------------------------------
train_data = datasets.ImageFolder(train_path, transform=transform)
test_data = datasets.ImageFolder(test_path, transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# ------------------------------
# ⑤ モデルを2種類用意：自作CNNとResNet18
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def create_resnet(num_classes):
    # ImageNetで学習済み重みを使う
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # model = resnet18(weights=ResNet18_Weights.DEFAULT) # 最新の公式重みで初期化
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ------------------------------
# ⑥ 学習関数
# ------------------------------
def train_model(model, loader, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    print(f"{name} の学習を開始します...")
    for epoch in range(2):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{name}] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# ------------------------------
# ⑦ 評価関数
# ------------------------------
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
    print(f"[{name}] 正答率: {100 * correct / total:.2f}%")

# ------------------------------
# ⑧ 実行部分
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自作CNNで学習・評価
simple_model = SimpleCNN(num_classes)
train_model(simple_model, train_loader, "自作CNN")
evaluate(simple_model, test_loader, "自作CNN")

# ResNet18で学習・評価
resnet_model = create_resnet(num_classes).to(device)
train_model(resnet_model, train_loader, "ResNet18")
evaluate(resnet_model, test_loader, "ResNet18")

# ------------------------------
# ⑨ 結果保存
# ------------------------------
torch.save(simple_model.state_dict(), f"results/{dataset_type}_simplecnn.pth")
torch.save(resnet_model.state_dict(), f"results/{dataset_type}_resnet18.pth")
print("モデル保存完了。お疲れさまでした！")
