"""Sobel 边缘检测核心算法（纯 numpy 实现）"""

import numpy as np


# Sobel 卷积核
SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float64)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对灰度图像进行 2D 卷积（zero-padding，输出尺寸与输入一致），使用 numpy 切片加速"""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # zero-padding 保持输出尺寸与输入相同
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    oh, ow = image.shape

    output = np.zeros((oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[i:i+oh, j:j+ow]
    return output


def sobel_edge(image: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """对灰度图像执行 Sobel 边缘检测。

    Args:
        image: 输入灰度图像（2D numpy 数组，值域 0-255）
        threshold: 可选阈值，低于此值的梯度置为 0。默认 None（不设阈值）

    Returns:
        边缘梯度幅值图（uint8，0-255）
    """
    if image.ndim != 2:
        raise ValueError("输入必须是灰度图像（2D 数组）")

    img = image.astype(np.float64)

    gx = _convolve2d(img, SOBEL_X)
    gy = _convolve2d(img, SOBEL_Y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # 归一化到 0-255
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0

    if threshold is not None:
        magnitude[magnitude < threshold] = 0

    return magnitude.astype(np.uint8)


def sobel_gradient_direction(image: np.ndarray) -> np.ndarray:
    """计算 Sobel 梯度方向（弧度）。

    Args:
        image: 输入灰度图像（2D numpy 数组）

    Returns:
        梯度方向角（弧度，范围 [-pi, pi]），与边缘幅值图同尺寸
    """
    if image.ndim != 2:
        raise ValueError("输入必须是灰度图像（2D 数组）")

    img = image.astype(np.float64)

    gx = _convolve2d(img, SOBEL_X)
    gy = _convolve2d(img, SOBEL_Y)

    return np.arctan2(gy, gx)


def classify_direction(angles: np.ndarray) -> np.ndarray:
    """将梯度方向角离散分类为 4 个方向。

    Args:
        angles: arctan2 返回的梯度方向角（弧度）

    Returns:
        分类标签数组：0=水平, 1=垂直, 2=对角线(+45°), 3=对角线(-45°)
    """
    # 将角度归一化到 [0, 180) 度范围
    deg = np.degrees(angles) % 180

    result = np.zeros_like(deg, dtype=np.int8)

    # 水平方向 (0° ± 22.5°)
    result[(deg >= 0) & (deg < 22.5)] = 0
    result[(deg >= 157.5) & (deg < 180)] = 0

    # 垂直方向 (90° ± 22.5°)
    result[(deg >= 67.5) & (deg < 112.5)] = 1

    # +45° 对角线
    result[(deg >= 22.5) & (deg < 67.5)] = 2

    # -45° 对角线
    result[(deg >= 112.5) & (deg < 157.5)] = 3

    return result
