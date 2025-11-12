# split_dataset.py
# 目的：rawフォルダ内のクラス別画像を、train/testフォルダに指定割合で振り分ける自動スクリプト

import os
import shutil
import random

def split_data(raw_dir, output_dir, train_ratio=0.8):
    """
    raw_dir: 元画像がクラス別フォルダにまとまったフォルダパス
    output_dir: 出力先の親フォルダ。ここにtrain/testフォルダが作られる
    train_ratio: 学習用に割り当てる割合(0.0~1.0)
    """

    # train/testのフォルダを作成
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # クラス名のリスト取得（rawフォルダ内のサブフォルダ名）
    classes = os.listdir(raw_dir)

    for cls in classes:
        cls_raw_path = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_raw_path):
            continue

        # 出力用クラスフォルダ作成
        cls_train_path = os.path.join(train_dir, cls)
        cls_test_path = os.path.join(test_dir, cls)
        os.makedirs(cls_train_path, exist_ok=True)
        os.makedirs(cls_test_path, exist_ok=True)

        # 画像ファイル一覧を取得しシャッフル
        images = [f for f in os.listdir(cls_raw_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        random.shuffle(images)

        # 分割数決定
        train_count = int(len(images) * train_ratio)

        # 画像ファイルをコピー（移動も可）
        for i, img_name in enumerate(images):
            src_path = os.path.join(cls_raw_path, img_name)
            if i < train_count:
                dst_path = os.path.join(cls_train_path, img_name)
            else:
                dst_path = os.path.join(cls_test_path, img_name)
            shutil.copy2(src_path, dst_path)  # ファイルコピー

        print(f"{cls} クラス：{len(images)}枚中 {train_count}枚をtrain, {len(images)-train_count}枚をtestに振り分け")

if __name__ == "__main__":
    # 実行例
    raw_folder = "../datasets/raw_original"    # 元画像がクラス別にまとまっているフォルダ
    output_folder = "../datasets/raw"          # train/testフォルダをここに作成
    split_data(raw_folder, output_folder, train_ratio=0.8)
    print("データセット分割が完了しました。")
