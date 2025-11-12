# 推論処理（単体画像クラス推定）
# 目的：学習済みモデルを使って単体の画像を分類する（良品・不良品、または車種判定）
# 画像ファイルを引数に受け取り、単体推論を行う
# 学習済みモデルを使って「1枚の画像を良品/不良品 or 車種」に分類するコード

import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN, build_resnet_model

# 評価値賞モデルの種類とデータセットを設定
dataset_type = "mvtec"      # "cars"に変更可能
model_type = "resnet"       # SimpleCNNに変更可能

# クラス名のリストを準備
if dataset_type == "mvtec":
    class_names = ["good", "defect"]
else:
    class_names = ["Acura_NSX", "BMW_M3", "Toyota_Corolla", "Other_Class"]  # 例：実際は196車種分ある

# GPUが使用できるか自動判定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modelをロード
if model_type == "simple":
    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(f"../results/{dataset_type}_simplecnn.pth", map_location=device))
else:
    model = build_resnet_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(f"../results/{dataset_type}_resnet18.pth", map_location=device))

model.to(device)
model.eval()

# 推論対象の画像ファイルを指定
image_path = "../datasets/sample/test_img.jpg"  # 推論したい画像を置く場所
img = Image.open(image_path).convert("RGB")

# 画像前処理（モデルに合うサイズに整形）
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Totensor(),
])
input_img = transform(img).unsqueeze(0).to(device)  # 1枚だけでもバッチ対応できるように整形

# modelに入力して結果を取得
with torch.no_grad():
    output = model(input_img)
    pred_idx = torch.argmax(output, 1).item()
    pred_class = class_names[pred_idx]

print(f"推論結果: {pred_class}")

# 結果を画像付きで可視化
import matplotlib.pyplot as plt
plt.imshow(img)
plt.title(f"Predicted: {pred_class}")
plt.axis("off")
plt.show