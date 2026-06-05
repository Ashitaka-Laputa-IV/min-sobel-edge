"""visualize.py 的单元测试"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt

from sobel_edge.visualize import compare_plot


class TestComparePlot:
    def test_returns_figure(self):
        """应返回 matplotlib Figure 对象"""
        original = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        edge = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        fig = compare_plot(original, edge, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_two_axes(self):
        """Figure 应包含两个子图"""
        original = np.zeros((30, 30), dtype=np.uint8)
        edge = np.zeros((30, 30), dtype=np.uint8)
        fig = compare_plot(original, edge, show=False)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_save_to_file(self, tmp_path):
        """应能保存到文件"""
        original = np.zeros((20, 20), dtype=np.uint8)
        edge = np.zeros((20, 20), dtype=np.uint8)
        path = tmp_path / "compare.png"
        fig = compare_plot(original, edge, save_path=str(path), show=False)
        assert path.exists()
        plt.close(fig)
