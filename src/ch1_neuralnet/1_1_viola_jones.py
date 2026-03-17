#%%
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_grayscale_image_as_numpy(image_path: str, out_size=(24, 24)) -> np.ndarray:
    """
    画像を読み込み、グレースケールのNumPy配列に変換する。

    Parameters
    ----------
    image_path : str
        入力画像パス
    out_size : tuple[int, int] | None
        出力サイズ (H, W)。Noneならリサイズしない。

    Returns
    -------
    img : np.ndarray
        shape=(H, W), dtype=np.float32
    """
    img = Image.open(image_path).convert("L")

    if out_size is not None:
        # PIL は (width, height) 指定
        img = img.resize((out_size[1], out_size[0]), Image.BILINEAR)

    img = np.asarray(img, dtype=np.float32)
    return img


def compute_integral_image(img: np.ndarray) -> np.ndarray:
    """
    Integral Image を自前実装で計算する。

    返す配列は 1行1列だけ大きくし、
    ii[y, x] = 元画像の [0:y, 0:x) の総和
    とする。

    これにより矩形和を簡潔に計算できる。

    Parameters
    ----------
    img : np.ndarray
        shape=(H, W)

    Returns
    -------
    ii : np.ndarray
        shape=(H+1, W+1)
    """
    h, w = img.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)

    # 2重累積和
    ii[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return ii


def rect_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Integral Image を使って矩形領域の総和を O(1) で計算する。

    矩形は左上 (x, y)、幅 w、高さ h。
    対象領域は [y:y+h, x:x+w]

    Returns
    -------
    s : float
        矩形内画素値合計
    """
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]


def haar_type_2_x(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    左右2分割のHaar-like特徴。
    feature = left_sum - right_sum

    条件: w は偶数
    """
    assert w % 2 == 0, "type-2-x requires even width"
    half_w = w // 2
    left = rect_sum(ii, x, y, half_w, h)
    right = rect_sum(ii, x + half_w, y, half_w, h)
    return left - right


def haar_type_2_y(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    上下2分割のHaar-like特徴。
    feature = top_sum - bottom_sum

    条件: h は偶数
    """
    assert h % 2 == 0, "type-2-y requires even height"
    half_h = h // 2
    top = rect_sum(ii, x, y, w, half_h)
    bottom = rect_sum(ii, x, y + half_h, w, half_h)
    return top - bottom


def enumerate_haar_features(
    image_shape: Tuple[int, int],
    feature_types: Tuple[str, ...] = ("type-2-x", "type-2-y"),
    min_feature_width: int = 2,
    min_feature_height: int = 2,
) -> List[Dict]:
    """
    指定画像サイズ上で使える Haar-like特徴の座標一覧を作る。

    Returns
    -------
    features : list[dict]
        各要素の例:
        {
            "type": "type-2-x",
            "x": 0,
            "y": 0,
            "w": 4,
            "h": 6,
        }
    """
    H, W = image_shape
    features = []

    for feat_type in feature_types:
        for h in range(min_feature_height, H + 1):
            for w in range(min_feature_width, W + 1):
                # feature type ごとの形状制約
                if feat_type == "type-2-x" and w % 2 != 0:
                    continue
                if feat_type == "type-2-y" and h % 2 != 0:
                    continue

                for y in range(0, H - h + 1):
                    for x in range(0, W - w + 1):
                        features.append(
                            {
                                "type": feat_type,
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                            }
                        )

    return features


def compute_haar_feature(ii: np.ndarray, feature: Dict) -> float:
    """
    1個の Haar-like特徴を計算する。
    """
    feat_type = feature["type"]
    x = feature["x"]
    y = feature["y"]
    w = feature["w"]
    h = feature["h"]

    if feat_type == "type-2-x":
        return haar_type_2_x(ii, x, y, w, h)
    elif feat_type == "type-2-y":
        return haar_type_2_y(ii, x, y, w, h)
    else:
        raise ValueError(f"Unsupported feature type: {feat_type}")


def extract_haar_features_manual(
    img: np.ndarray,
    feature_defs: List[Dict],
) -> np.ndarray:
    """
    画像1枚から、自前実装で Haar-like特徴ベクトルを抽出する。

    Returns
    -------
    feats : np.ndarray
        shape=(num_features,), dtype=np.float32
    """
    ii = compute_integral_image(img)
    feats = np.array(
        [compute_haar_feature(ii, f) for f in feature_defs],
        dtype=np.float32,
    )
    return feats

def visualize_haar_feature(img, feature, feature_value=None):
    """
    Haar-like特徴を可視化する

    Parameters
    ----------
    img : np.ndarray
        グレースケール画像 (H,W)

    feature : dict
        {
            "type": "type-2-x",
            "x": int,
            "y": int,
            "w": int,
            "h": int
        }

    feature_value : float | None
        計算された特徴値
    """

    x = feature["x"]
    y = feature["y"]
    w = feature["w"]
    h = feature["h"]
    ftype = feature["type"]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # ------------------------------
    # 元画像
    # ------------------------------

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("image")
    ax[0].axis("off")

    # ------------------------------
    # フィルタ領域
    # ------------------------------

    ax[1].imshow(img, cmap="gray")
    ax[1].set_title(f"Haar feature ({ftype})")

    if ftype == "type-2-x":

        half = w // 2

        # 白領域
        ax[1].add_patch(
            Rectangle(
                (x, y),
                half,
                h,
                edgecolor="red",
                facecolor="red",
                alpha=0.3,
            )
        )

        # 黒領域
        ax[1].add_patch(
            Rectangle(
                (x + half, y),
                half,
                h,
                edgecolor="blue",
                facecolor="blue",
                alpha=0.3,
            )
        )

    elif ftype == "type-2-y":

        half = h // 2

        ax[1].add_patch(
            Rectangle(
                (x, y),
                w,
                half,
                edgecolor="red",
                facecolor="red",
                alpha=0.3,
            )
        )

        ax[1].add_patch(
            Rectangle(
                (x, y + half),
                w,
                half,
                edgecolor="blue",
                facecolor="blue",
                alpha=0.3,
            )
        )

    if feature_value is not None:
        ax[1].text(
            0.02,
            0.95,
            f"value={feature_value:.2f}",
            transform=ax[1].transAxes,
            color="yellow",
            fontsize=12,
            verticalalignment="top",
        )

    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_filter_shape(feature, size=64):
    """
    Haarフィルタ形状だけを表示
    """

    ftype = feature["type"]
    w = feature["w"]
    h = feature["h"]

    canvas = np.zeros((h, w))

    if ftype == "type-2-x":
        half = w // 2
        canvas[:, :half] = 1
        canvas[:, half:] = -1

    elif ftype == "type-2-y":
        half = h // 2
        canvas[:half, :] = 1
        canvas[half:, :] = -1

    plt.imshow(canvas, cmap="bwr")
    plt.colorbar()
    plt.title(f"filter shape ({ftype})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    image_path = "./../../data/cmu_face/an2i_straight_neutral_open.jpg"

    img = load_grayscale_image_as_numpy(image_path, out_size=(24, 24))
    print("img shape:", img.shape)
    print("dtype:", img.dtype)
    print("min/max:", img.min(), img.max())

    feature_defs = enumerate_haar_features(
        image_shape=img.shape,
        feature_types=("type-2-x", "type-2-y"),
        min_feature_width=2,
        min_feature_height=2,
    )

    print("num feature defs:", len(feature_defs))
    # 画像から特徴ベクトルを抽出
    feats = extract_haar_features_manual(img, feature_defs)
    print("feature vector shape:", feats.shape)
    print("first 10 features:", feats[:10])

    ###### 可視化 ######
    # 適当に1つ選ぶ
    feature = feature_defs[501]

    # 特徴値計算
    ii = compute_integral_image(img)
    value = compute_haar_feature(ii, feature)

    # 可視化
    visualize_haar_feature(img, feature, value)
    visualize_filter_shape(feature)

# %%
