# # データセットクラスを定義（フォルダ読み込み・ラベル付与など）
# 目的：画像フォルダの中身を読み込み、モデル学習用に整理して渡すクラス定義

import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """
    自作の画像データセットクラス。
    画像フォルダのクラス別サブフォルダから画像とラベルを読み込む。
    """

    def __init__(self, root_dir, transform=None):
        """
        root_dir: 画像がクラス別フォルダに分かれてある親フォルダのパス
        transform: 画像前処理（transform.Composeなど）を受け取る
        """
        self.root_dir = root_dir
        self.transform = transform

        # フォルダ内のクラス名（サブフォルダ名）を一覧取得し、ラベル付け
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 画像ファイルのパスとラベルを記録するためのリストを用意
        self.image_paths = []
        self.labels = []

        # 各クラスフォルダを巡回し、画像ファイルパスを取得
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                # 画像ファイルとみなしてパスをリストに追加
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(cls_folder, fname))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        # データセット全体の画像数を返す
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        インデックス指定で1枚の画像とラベルを返す。
        """
        # PIL.Imageで画像を開く
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        # 画像が前処理(transform)を持っていたら適用
        if self.transform:
            image = self.transform(image)

        return image, label
