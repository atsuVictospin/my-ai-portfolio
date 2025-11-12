# 画像前処理、データ拡張の関数群
# 目的：モデル学習のための画像前処理・拡張関数をまとめる

from torchvision import transforms

def get_default_transforms():
    """
    画像サイズの変更・テンソル変換など基本的な前処理をまとめる
    """
    return transforms.Compose([
        transforms.Resize((128, 128)),    # 画像を128×128にリサイズ
        transforms.ToTensor(),            # 画像をPyTorchのテンソル型に変換
        # 画素値を0~1に正規化（RGB各チャネル毎に平均・分散を適用して標準化）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_train_transforms():
    """
    学習時用の前処理・データ拡張セット。
    ランダムな回転や反転など、多様性を持たせる。
    """
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),      # 水平反転をランダムで行う
        transforms.RandomRotation(15),         # ±15度ランダム回転
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    """
    テスト・推論時用の前処理セット。
    学習時の強い拡張はかけない（回転や反転なし)
    """
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
