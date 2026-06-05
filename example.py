from sobel_edge import load_image, sobel_edge, sobel_gradient_direction, classify_direction, compare_plot, save_image

# 加载图片（支持文件路径或 numpy 数组）
img = load_image("stones.png")

# 基础边缘检测
edge = sobel_edge(img)

# 带阈值的边缘检测
edge_filtered = sobel_edge(img, threshold=100)

# 梯度方向（弧度）
angles = sobel_gradient_direction(img)

# 方向分类：0=水平, 1=垂直, 2=+45°对角线, 3=-45°对角线
directions = classify_direction(angles)

# 可视化对比（原图 vs 边缘图）
compare_plot(img, edge, show=True)