# 評価処理（精度算出・混同行列表示）
# 学習済みモデルを読み込み、テストデータで評価し、正答率や混合行列で性能を評価する

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from model import SimpleCNN, build_resnet_model


# GPUが使用できるか確認（CPUのみのときも自動で対応）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 評価するデータセットの種類を設定
datasets_type = "mvtec"   # "cars"にするとstanford_carsを使用可能
test_path = f"../datasets/raw/{datasets_type}_ad/val" if datasets_type == "mvtec" else f"datasets/raw/stanford_cars/test"

# データの前処理（画像サイズを64x64に統一）
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# データをフォルダから読み込み
test_data = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
class_names = test_data.classes

# モデルの種類を選ぶ("SimpleCNN or build_resnet_model")
model_type = "resnet"

if model_type == "Simple":
    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(f"../results/{datasets_type}_simplecnn.pth", map_location=device))
else:
    model = build_resnet_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(f"../results/{datasets_type}_resnet18.pth", map_location=device))

model.to(device)
model.eval()

# 評価処理
all_preds, all_labels = [], []
with torch.no_grad():     # 推論時は勾配計算を省略化して高速化
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 正答率（Accuracy）を計算
accuracy = accuracy_score(all_labels, all_preds)
print(f"正答率: {accuracy * 100:.2f}%")

# 混合行列を生成
cm = confusion_matrix(all_labels, all_preds)

# 混合行列をヒートマップで表示
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("予測クラス")
plt.ylabel("正解クラス")
plt.title(f"Confusion Matrix ({datasets_type.upper()} - {model_type})")
plt.savefig(f"../results/confusion_matrix_{datasets_type}_{model_type}.png")
plt.show

print("混合行列を resultsフォルダに保存しました！")