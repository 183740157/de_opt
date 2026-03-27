#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：de_opt
@File    ：algo_const.py
@IDE     ：PyCharm
@Author  ：wubc
@Date    ：2024/10/15 19:06
描述      ：算法相关常数
"""

from opfunu.cec_based.cec2014 import *
from opfunu.cec_based.cec2017 import *

# 差分进化通用常数
NP = 100  # 种群大小
D = 10  # 维度
MAX_FES = 5000 * D  # 最大函数评估次数
MAX_CYCLE = MAX_FES // NP  # 最大迭代轮数
TRAIN_RUNTIME = 5  # 重复运行次数
TEST_RUNTIME = 30
LB = -100  # 下界
UB = 100  # 上界
LS_P = 0.20  # 每个精英触发局部搜索的概率
RECORD_N = 100  # 迭代每 n 轮循环记录一次最优效果

# 随机数种子
SEED = 23

# 训练数据集：CEC'14 测试套件的所有28个函数
# TRAIN_FUNCS = [F12014, F22014, F32014, F42014, F52014, F62014, F72014, F82014, F92014, F102014, F112014, F122014,
#                F132014, F142014, F152014, F162014, F172014, F182014, F192014, F202014, F212014, F222014, F232014,
#                F242014, F252014, F262014, F272014, F282014, F292014, F302014]
TRAIN_FUNCS = [F12014, F22014, F32014, F42014, F52014, F62014, F72014, F82014, F92014, F102014, F112014, F132014,
               F142014, F152014, F162014, F172014, F182014, F192014, F202014, F212014, F222014, F232014, F242014,
               F252014, F262014, F272014, F282014, F292014, F302014]

# 测试数据集：CEC'17 测试套件的所有29个函数
TEST_FUNCS = [F12017, F22017, F32017, F42017, F52017, F62017, F72017, F82017, F92017, F102017, F112017, F122017,
              F132017, F142017, F152017, F162017, F172017, F182017, F192017, F202017, F212017, F222017, F232017,
              F242017, F252017, F262017, F272017, F282017, F292017]
# TEST_FUNCS = [F12017, F92017, F282017, F292017]

# 算法效果输出目录
BASE_RES_DIR = '/result/de_base/'
REF_RES_DIR = '/result/de_reference/'
DE_RL_DNN_RES_DIR = '/result/de_rl_dnn/'
WBC_RES_DIR = '/result/de_wbc/'

# 差分进化算法多少次计算，强化学习迭代一次
DE_PER_RL_UPDATE = 10
