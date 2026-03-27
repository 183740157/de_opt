#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : de_reference.py
@Time    : 2025/7/14 14:05
@Author  : wubch
@Desc    : 差分进化算法，对照组
"""

import pyade
import pandas as pd
from tqdm import tqdm
from constants.algo_const import *
from constants.proj_const import *
from utils.time_util import *

# 算法参数
NP = 100  # 种群大小
D = 10  # 维度
MAX_FES = 200000  # 最大函数评估次数
MAX_CYCLE = MAX_FES // NP  # 最大迭代次数
TEST_RUNTIME = 30

from opfunu.cec_based.cec2022 import *

TEST_FUNCS = [F12022, F22022, F32022, F42022, F52022, F62022, F72022, F82022, F92022, F102022, F112022, F122022]

# 生成文档的配置
name = 'reference'

# 算法列表
algorithms = {
    'SaDE': __import__('pyade.sade', fromlist=['*']),
    'SHADE': __import__('pyade.shade', fromlist=['*']),
    'ILSHADE': __import__('pyade.ilshade', fromlist=['*']),
    'JSO': __import__('pyade.jso', fromlist=['*']),
    'MPEDE': __import__('pyade.mpede', fromlist=['*']),
    # 添加 LSHADE
    'LSHADE': __import__('pyade.lshade', fromlist=['*']),
}

# 结果文件准备
now_time = get_now_time('%Y%m%d_%H%M')
file_path = RESOURCE_PATH + REF_RES_DIR + f'de_{name}_{D}_{now_time}.csv'
pd.DataFrame(columns=['Algorithm', 'Function', 'Mean', 'Std', 'Convergence']).to_csv(
    file_path, index=False, mode='w')

for alg_name, alg_mod in algorithms.items():
    start_time = get_now_second()
    print(f"\n>>> Running {alg_name} ({TEST_RUNTIME} runs each) ...")
    for func_obj in tqdm(TEST_FUNCS, desc=alg_name):
        func = func_obj(ndim=D)
        raw_name = func_obj.name
        func_name = raw_name.split(":")[0]
        print(f'######## {alg_name}-{func_name} 开始 ########')

        bests = []
        all_conv_dicts = []

        for run in range(TEST_RUNTIME):
            conv_dict = {}


            def callback(**kwargs):
                fe = kwargs.get('current_generation') + 1
                best = min(kwargs.get('fitness'))
                if fe is not None and (fe == 1 or fe % RECORD_N == 0):
                    conv_dict[fe] = best


            params = alg_mod.get_default_params(dim=D)
            params['bounds'] = np.array([[LB, UB]] * D)
            params['func'] = func.evaluate
            params['max_evals'] = MAX_FES
            params['population_size'] = NP
            params['seed'] = SEED * run
            params['callback'] = callback

            _, best_f = alg_mod.apply(**params)
            bests.append(best_f)
            all_conv_dicts.append(conv_dict)
            print(f'{run + 1}.run: {best_f:.3e}')

        # 计算统计量
        mean_best = round(np.mean(bests), 3)
        std_best = round(np.std(bests), 3)

        # 收集所有 run 记录的 FE
        all_fes = sorted({fe for d in all_conv_dicts for fe in d.keys()})

        # 按 FE 对齐求平均
        mean_curve = []
        for fe in all_fes:
            vals = [d.get(fe, np.nan) for d in all_conv_dicts]
            mean_curve.append(round(np.nanmean(vals), 3))

        # 拼成字符串 fe:val;fe:val;...
        convergence_str = ";".join([f"{fe}:{v}" for fe, v in zip(all_fes, mean_curve)])

        print(f'Means of {TEST_RUNTIME} runs: {mean_best:.3e}')
        print(f'Std of {TEST_RUNTIME} runs: {std_best:.3e}')
        print(f'运行耗时 {get_now_second() - start_time} 秒\n')

        # 实时写入
        pd.DataFrame([{'Algorithm': alg_name, 'Function': func_name, 'Mean': mean_best, 'Std': std_best,
                       'Convergence': convergence_str}]).to_csv(file_path, index=False, mode='a', header=False)

print(f"\n全部完成，结果保存为 {file_path}")
