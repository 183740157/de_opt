#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：de_opt
@File    ：functions_show.py
@IDE     ：PyCharm
@Author  ：wubc
@Date    ：2024/04/30 13:22
描述      ：展示CEC数据集
"""

from constants.algo_const import TRAIN_FUNCS, TEST_FUNCS

# 遍历训练数据集并输出方法名
print("训练数据集方法名：")
for func in TRAIN_FUNCS:
    func_name = func.name.replace("’s", "")
    print(func_name, end=' ; ')

# 遍历测试数据集并输出方法名
print("\n测试数据集方法名：")
for func in TEST_FUNCS:
    func_name = func.name.replace("’s", "")
    print(func_name, end=' ; ')
