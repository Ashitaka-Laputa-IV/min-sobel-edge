"""图像输入输出模块"""

import numpy as np
from PIL import Image
from pathlib import Path


def load_image(source: str | Path | np.ndarray) -> np.ndarray:
    """加载图像为灰度 numpy 数组。

    Args:
        source: 图片文件路径或 numpy 数组。
                如果是 numpy 数组，2D 直接返回，3D 彩色图会转为灰度。

    Returns:
        灰度图像（2D uint8 数组，0-255）
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            # 使用 round 避免浮点截断（如 12.7 → 13 而非 12）
            return np.round(source).astype(np.uint8)
        if source.ndim == 3:
            # 彩色转灰度：使用 PIL
            return np.array(Image.fromarray(source).convert("L"), dtype=np.uint8)
        raise ValueError(f"不支持的数组维度: {source.ndim}，期望 2 或 3")

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {path}")

    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """将 numpy 数组保存为图片文件。

    Args:
        image: 灰度图像（2D uint8 数组）
        path: 保存路径

    Returns:
        保存的文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(image, mode="L")
    img.save(path)
    return path
