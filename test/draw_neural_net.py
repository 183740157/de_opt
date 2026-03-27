#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : draw_neural_net.py
@Time    : 2025/11/13 09:16
@Author  : wubch
@Desc    : 
"""

import matplotlib.pyplot as plt
import numpy as np


def draw_neural_net(input_dim, hidden_dim, output_dim):
    fig, ax = plt.subplots(figsize=(10, 8))

    # 设置节点的水平和垂直间距
    layer_spacing = 2
    node_spacing = 0.5

    # 绘制输入层
    input_x = 0
    input_nodes = [f'Input {i}' for i in range(input_dim)]
    input_y = np.linspace(0, hidden_dim * node_spacing, input_dim)
    for i, y in enumerate(input_y):
        ax.text(input_x, y, input_nodes[i], ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='skyblue', alpha=0.5))

    # 绘制隐藏层1
    hidden1_x = input_x + layer_spacing
    hidden1_nodes = [f'Hidden1 {i}' for i in range(hidden_dim)]
    hidden1_y = np.linspace(0, hidden_dim * node_spacing, hidden_dim)
    for i, y in enumerate(hidden1_y):
        ax.text(hidden1_x, y, hidden1_nodes[i], ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='lightgreen', alpha=0.5))

    # 绘制隐藏层2
    hidden2_x = hidden1_x + layer_spacing
    hidden2_nodes = [f'Hidden2 {i}' for i in range(hidden_dim)]
    hidden2_y = np.linspace(0, hidden_dim * node_spacing, hidden_dim)
    for i, y in enumerate(hidden2_y):
        ax.text(hidden2_x, y, hidden2_nodes[i], ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='lightgreen', alpha=0.5))

    # 绘制输出层
    output_x = hidden2_x + layer_spacing
    output_nodes = ['mu_F', 'mu_CR', 'mu_p', 'mu_Fw', 'mu_archF', 'mu_archP']
    output_y = np.linspace(0, output_dim * node_spacing, output_dim)
    for i, y in enumerate(output_y):
        ax.text(output_x, y, output_nodes[i], ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='lightcoral', alpha=0.5))

    # 绘制连接
    # 输入层到隐藏层1
    for i in range(input_dim):
        for j in range(hidden_dim):
            ax.plot([input_x, hidden1_x], [input_y[i], hidden1_y[j]], color='gray', alpha=0.5)

    # 隐藏层1到隐藏层2（带残差连接）
    for i in range(hidden_dim):
        ax.plot([hidden1_x, hidden2_x], [hidden1_y[i], hidden2_y[i]], color='gray', alpha=0.5)
        ax.plot([hidden1_x, hidden2_x], [hidden1_y[i], hidden2_y[i]], color='blue', alpha=0.5, linestyle='--')

    # 隐藏层2到输出层
    for i in range(hidden_dim):
        for j in range(output_dim):
            ax.plot([hidden2_x, output_x], [hidden2_y[i], output_y[j]], color='gray', alpha=0.5)

    # 设置图形属性
    ax.set_xlim(-1, output_x + 1)
    ax.set_ylim(-1, max(hidden_dim, output_dim) * node_spacing + 1)
    ax.axis('off')
    ax.set_title("Policy Network Structure", fontsize=16)

    plt.show()


if __name__ == "__main__":
    # 调用函数绘制网络结构图
    draw_neural_net(input_dim=10, hidden_dim=64, output_dim=6)
