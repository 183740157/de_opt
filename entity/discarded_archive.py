#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : discarded_archive.py
@Time    : 2025/8/6 16:24
@Author  : wubch
@Desc    : 
"""

import numpy as np


class DiscardedArchive:
    """
    安全地收集被淘汰个体的外部档案类，带类型/维度/边界验证。
    """

    def __init__(self, dim: int, max_cnt: int):

        self.dim = dim
        self.max_cnt = max_cnt
        self._data = []  # 真正存储的列表

    # -----------------------------
    # 对外表现像一个 list
    # -----------------------------
    def append(self, individual):
        """添加单个被淘汰个体"""

        if len(self._data) < self.max_cnt:
            self._data.append(individual)
            return

        # 最大最小距离替换
        data = np.array(self._data)
        # 新个体到所有现有个体的最小距离
        new_min_dist = np.min(np.linalg.norm(data - individual, axis=1))

        # 现有档案中每个个体的最小距离
        dist_matrix = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
        # 将对角线设为inf，避免自距离为0
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = np.min(dist_matrix, axis=1)
        # 找到最小距离的两个点的索引
        min_i, min_j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # 随机选择其中一个作为被替换的索引
        victim_idx = np.random.choice([min_i, min_j])

        # 如果新个体更分散则替换
        if new_min_dist > min_dists[victim_idx]:
            self._data[victim_idx] = individual

    def extend(self, individuals):
        """批量添加"""
        for ind in individuals:
            self.append(ind)

    def clear(self):
        """清空档案"""
        self._data.clear()

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def to_list(self):
        return self._data.copy()
