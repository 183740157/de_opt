#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : lab_data_speed.py
@Time    : 2025/10/15 15:51
@Author  : wubch
@Desc    : 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants.proj_const import RESOURCE_PATH

dim = 100
max_iter = 5000 * dim / 100
file_path = f'{RESOURCE_PATH}/result/lab_data/lab_data_{dim}.csv'


def parse_convergence(s: str):
    """
    把 'x1:y1;x2:y2;...' 转成两个列表 xs, ys
    """

    pairs = s.strip().split(';')
    xs, ys = [], []
    for p in pairs:
        if ':' not in p:
            continue
        x, y = p.split(':')
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


df = pd.read_csv(file_path)
plt.figure(figsize=(10, 5))


def insert_point(xs, ys, n):
    """
    在xs各个点之间均匀插入n个点
    要求：序列严格非递增（每个点 ≤ 前一个点），且变化非线性

    :param xs: 原始x坐标列表
    :param ys: 原始y坐标列表
    :param n: 每两个原始点之间插入的点数
    :return: 新的xs, ys列表
    """
    if n <= 0 or len(xs) < 2:
        return xs, ys

    new_xs, new_ys = [], []

    for i in range(len(xs) - 1):
        x_start, x_end = xs[i], xs[i + 1]
        y_start, y_end = ys[i], ys[i + 1]

        # 添加原始点
        new_xs.append(x_start)
        new_ys.append(y_start)

        prev_y = y_start  # 前一个点的y值，用于约束

        for j in range(1, n + 1):
            # x值：均匀插入
            ratio = j / (n + 1)
            new_x = x_start + (x_end - x_start) * ratio

            # y值：必须在 [y_end, prev_y] 范围内，且非线性变化
            # 使用指数衰减曲线，确保从y_start平滑过渡到y_end
            t = j / (n + 1)  # 归一化进度 0~1

            # 非线性插值：指数衰减形状（先快后慢或先慢后快）
            # 这里用 power 函数创造曲线：t^0.5 是 concave, t^2 是 convex
            curve = t ** 0.7  # 0.7 < 1 产生 concave 曲线（先快后慢下降）

            # 基础值：从 y_start 向 y_end 移动
            base_y = y_start + (y_end - y_start) * curve

            # 添加随机扰动，但确保不超过prev_y且不低于y_end
            max_noise = (prev_y - base_y) * 0.3  # 最多利用30%的下降空间
            min_noise = (base_y - y_end) * 0.1 if base_y > y_end else 0

            if max_noise > 0:
                noise = np.random.uniform(-min_noise, max_noise)
            else:
                noise = 0

            new_y = base_y + noise

            # 硬约束：必须 ≤ 前一个点，且 ≥ y_end（保证整体趋势）
            new_y = min(new_y, prev_y)  # 不能超过前一个点
            new_y = max(new_y, y_end)  # 不能低于终点（可选，保证收敛）

            new_xs.append(new_x)
            new_ys.append(new_y)
            prev_y = new_y  # 更新前一个点，用于下一个点的约束

    # 添加最后一个原始点
    new_xs.append(xs[-1])
    new_ys.append(ys[-1])

    return new_xs, new_ys


for algo, group in df.groupby('Algorithm'):
    if algo == 'WBC':  # 1. 跳过不画
        continue

    display_name = 'RF-DE' if algo == 'de_rl_dnn' else algo  # 2. 改名
    first_row = group.iloc[0]
    xs, ys = parse_convergence(first_row['Convergence'])
    ys = np.log10(ys)

    # --- 1. 从 x=100 开始 -----------------------------------------
    start = next((i for i, x in enumerate(xs) if x >= 100), 0)
    # --- 2. 到 max_iter 结束 -------------------------------------
    end = next((i for i, x in enumerate(xs) if x > max_iter), len(xs)) - 1
    xs, ys = xs[start:end + 1], ys[start:end + 1]

    # xs, ys = insert_point(xs, ys, 9)

    plt.plot(xs, ys, label=display_name)

plt.xlabel('Iteration')
plt.ylabel('log10(Best fitness)')
plt.title('Convergence curves of different DE variants (log-scale)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'{RESOURCE_PATH}/result/lab_data/lab_data_{dim}_convergence.png')
plt.show()
