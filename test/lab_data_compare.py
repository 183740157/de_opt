#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : lab_data_compare.py
@Time    : 2025/8/20 16:29
@Author  : wubch
@Desc    : Wilcoxon signed-rank test
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from collections import defaultdict
from constants.proj_const import RESOURCE_PATH
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from tabulate import tabulate

# dims = [10, 30, 50, 100]
dims = [10, 20]

# 定义固定的算法顺序
# ALGO_ORDER = ['DE', 'ILSHADE', 'JSO', 'LSHADE', 'MPEDE', 'SHADE', 'SaDE', 'RL-DE']
ALGO_ORDER = ['ILSHADE', 'JSO', 'LSHADE', 'MPEDE', 'SHADE', 'SaDE', 'RL-DE']


def print_dict():
    for d, algo_dict in data_dict.items():
        print(f'Dim={d}, 示例:')
        for algo, func_map in list(algo_dict.items()):
            print(f'  {algo}: {dict(list(func_map.items())[:5])}')


def friedman_wilcoxon_for_dim(data_dict, dim):
    """
    data_dict[dim] = {算法名: {函数名: (平均值, 标准差)}}
    返回: friedman_p, mean_rank_dict, wilcoxon_df
    """

    # 1. 拉成 DataFrame：行=函数，列=算法，值=平均值
    records = []
    algos = list(data_dict[dim].keys())
    funcs = list(next(iter(data_dict[dim].values())).keys())

    for func in funcs:
        row = [data_dict[dim][algo][func][0] for algo in algos]
        records.append(row)
    df = pd.DataFrame(records, columns=algos, index=funcs)

    # 2. Friedman 检验
    stat, p_fried = friedmanchisquare(*[df[col] for col in algos])

    # 3. 计算平均秩
    ranks = df.rank(axis=1, ascending=True)  # 越小越好
    mean_rank = ranks.mean(axis=0).sort_values()

    # 4. Wilcoxon 两两检验
    wilcox_results = []
    for a1, a2 in combinations(algos, 2):
        w, p = wilcoxon(df[a1], df[a2], alternative='two-sided')
        wilcox_results.append((a1, a2, p))
    wilcox_df = pd.DataFrame(wilcox_results, columns=['Algo1', 'Algo2', 'p_value'])

    return p_fried, mean_rank.to_dict(), wilcox_df


def wilcoxon_effect_size(x, y):
    """Wilcoxon 匹配对秩效应量 r"""
    from scipy.stats import wilcoxon
    w, _ = wilcoxon(x, y, alternative='two-sided')
    n = len(x)
    r = abs(w - n * (n + 1) / 4) / (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    return r


if __name__ == "__main__":
    # 目标结构：{维度: {算法名: {函数名: (平均值, 标准差)}}}
    data_dict = defaultdict(lambda: defaultdict(dict))

    for dim in dims:
        # file_path = f'{RESOURCE_PATH}/result/lab_data/lab_data_{dim}.csv'
        file_path = f'{RESOURCE_PATH}/result/lab_data/lab_data_{dim}_2022.csv'
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                algo = row['Algorithm'].strip()
                func = row['Function'].strip()
                mean = float(row['Mean'])
                std = float(row['Std'])
                # 填充三层结构
                data_dict[dim][algo][func] = (mean, std)

    data_dict = {d: dict(inner) for d, inner in data_dict.items()}
    # print_dict()

    for dim in dims:
        p, rank_dict, wilcox_df = friedman_wilcoxon_for_dim(data_dict, dim)

        # ---- 打印 Friedman ----
        print(f"\n==== Dim={dim}  Friedman Test  ====")
        print(f"p-value = {p:.4e}   (越小越显著)")
        print("平均秩（越小越好）：")
        for algo, r in rank_dict.items():
            print(f"  {algo:10s}: {r:.2f}")

        # ---- 打印 Wilcoxon 矩阵 ----
        print("\n两两 Wilcoxon p-value 矩阵（<0.05 可视为显著差异）")
        # 转成矩阵形式方便查看
        pivot = wilcox_df.pivot(index='Algo1', columns='Algo2', values='p_value')
        pivot = pivot.combine_first(pivot.T).fillna(1.0)

        # 按照指定顺序重新排列行列
        pivot = pivot.reindex(index=ALGO_ORDER, columns=ALGO_ORDER)

        # 确保RL-DE行显示所有值（包括对角线）
        for col in pivot.columns:
            if pd.isna(pivot.loc['RL-DE', col]):
                if col == 'RL-DE':
                    pivot.loc['RL-DE', col] = 1.0
                else:
                    # 从对称位置获取
                    pivot.loc['RL-DE', col] = pivot.loc[col, 'RL-DE']

        print(tabulate(pivot, headers='keys', tablefmt='psql', floatfmt='.2e'))
