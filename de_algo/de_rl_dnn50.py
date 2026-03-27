#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：de_opt
@File    ：de_rl_dnn.py
@IDE     ：PyCharm
@Author  ：wubc
@Date    ：2025/04/27 16:10
描述      ：策略梯度算法与DE算法集成——全连接神经网络
"""

import os
import random
import torch
import pandas as pd
from neural_networks.de_rl_dnn_network import PolicyGradient
from constants.algo_const import *
from constants.proj_const import *
from utils.time_util import *
from entity.discarded_archive import DiscardedArchive
from scipy.optimize import minimize
from numba import njit

# 设置随机数种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NP = 100  # 种群大小
D = 50  # 维度
MAX_FES = 5000 * D  # 最大函数评估次数
MAX_CYCLE = MAX_FES // NP  # 最大迭代次数
TRAIN_RUNTIME = 5  # 重复运行次数
TEST_RUNTIME = 30


# ---------- 与 DE 相关的工具函数 ----------
def initial(func):
    """返回初始种群及其适应度"""

    X = [func.create_solution() for _ in range(NP)]
    f = [func.evaluate(x) for x in X]
    return X, f


def de_mutation(X, f, F, Fw, p, archF, archP, discarded):
    """
    向量化实现：
    DE/current-to-pBest-w/1 + Archive/1
    返回 (NP, D) 的 ndarray
    """

    X = np.asarray(X, dtype=float)
    NP, D = X.shape

    # 1) pBest 候选索引
    sorted_idx = np.argsort(f)
    pBest_cnt = max(1, int(p * NP))
    pBest_idx = sorted_idx[:pBest_cnt]

    # 2) 为每个个体 i 随机选一个 pBest
    pBest_sel = np.random.choice(pBest_idx, size=NP)

    # 3) 为每个个体 i 随机选两个不同且不等于 i 的 r1, r2
    rng = np.arange(NP)
    r1 = np.empty(NP, dtype=int)
    r2 = np.empty(NP, dtype=int)
    for i in range(NP):
        others = np.delete(rng, i)
        r1[i], r2[i] = np.random.choice(others, 2, replace=False)

    # 4) 基础差分向量
    V = X + Fw * (X[pBest_sel] - X) + F * (X[r1] - X[r2])

    # 5) archive 成分（按概率叠加）
    if len(discarded) >= 1 and np.random.rand() < archP:  # 档案≥1即可
        archive = np.asarray(discarded.to_list(), dtype=float)
        # 从前 pBest_cnt 中随机挑一个
        pBest_pick = np.random.choice(pBest_idx, size=NP)
        # 从档案随机挑一个
        a_idx = np.random.randint(0, len(archive), size=NP)
        a_vec = archive[a_idx]
        # 单一扰动
        V += archF * (X[pBest_pick] - a_vec)

    # 6) 边界裁剪
    return reflect_clip(V, LB, UB)


def de_crossover(X, V, CR):
    """
    向量化二项交叉
    输入: X, V 可以是 list[list[float]] 或 ndarray
    输出: (NP, D) ndarray
    """

    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    NP, D = X.shape

    # 1) 每行随机选一个维度 k 必交叉
    k = np.random.randint(0, D, size=NP).reshape(-1, 1)

    # 2) 随机掩码
    rand = np.random.rand(NP, D)
    mask = (rand <= CR) | (np.arange(D) == k)  # True → 用 V, False → 用 X

    # 3) 交叉
    U = np.where(mask, V, X)
    return U


def archive_save_prob(it, p_min=0.01, p_max=1.0, decay=0.98):
    """
    随迭代次数指数衰减的存档概率
    p(it) = p_min + (p_max - p_min) * decay^it
    """

    return max(p_min, p_min + (p_max - p_min) * (decay ** it))


def de_selection(func, X, f, U, discarded, it):
    """选择：根据贪婪策略更新种群，并保存被淘汰的个体"""

    prob = archive_save_prob(it)
    for i in range(NP):
        obj_u = func.evaluate(U[i])
        if obj_u < f[i]:
            # 按概率存档
            if random.random() < prob:
                discarded.append(X[i])

            # 替换
            X[i] = U[i].copy()
            f[i] = obj_u
    return X, f, discarded


@njit(fastmath=True)
def reflect_clip(V, LB, UB):
    """
    镜像反射边界修正（mirror / bounce）
    参数:
        V  : ndarray (NP, D)
        LB : float 下界
        UB : float 上界
    返回:
        ndarray (NP, D)
    """

    span = UB - LB
    # 先把值折进 [LB, UB] 的周期区间
    offset = (V - LB) % (2 * span)
    # 再镜像到正确区间
    V = LB + np.where(offset > span, 2 * span - offset, offset)
    return V


def log_scale(value):
    """缩放"""
    return np.sign(value) * np.log10(np.abs(value) + 1)


# ---------- 状态构建 ----------
def get_state(f, it):
    """构建强化学习的状态向量"""

    f_log = log_scale(f)
    min_f = np.min(f_log)
    q1_f = np.percentile(f_log, 25)
    median_f = np.median(f_log)
    q3_f = np.percentile(f_log, 75)
    max_f = np.max(f_log)
    mean_f = np.mean(f_log)
    std_f = np.std(f_log)

    median_f = median_f if median_f != 0 else 1e-8
    min_f /= median_f
    q1_f /= median_f
    median_f = 1
    q3_f /= median_f
    max_f /= median_f
    mean_f /= median_f
    std_f /= median_f

    progress = (it + 1) / MAX_CYCLE
    log_progress = np.log10(progress)

    state = np.array([min_f, q1_f, median_f, q3_f, max_f, mean_f, std_f, log_progress], dtype=np.float32)
    return state


def refine_best_with_lbfgs(func, best_x, max_eval=D):
    """
    用 L-BFGS-B 对当前最优个体做局部精修
    返回：新解、新适应度、实际消耗的函数评估次数
    """
    eval_cnt = [0]  # 闭包计数器

    def _f(x):
        eval_cnt[0] += 1
        return func.evaluate(x)

    res = minimize(
        _f,
        best_x,
        method='L-BFGS-B',
        bounds=list(zip([LB] * len(best_x), [UB] * len(best_x))),
        options={'maxiter': max_eval}
    )
    return res.x, res.fun, eval_cnt[0]


# ---------- 训练 ----------
def de_policy_gradient_train_main():
    now_time = get_now_time('%Y%m%d_%H%M')
    main_start_time = get_now_second()
    file_path = f'{RESOURCE_PATH}{DE_RL_DNN_RES_DIR}de_rl_dnn_train_{D}_{now_time}.csv'
    model_path = f'{RESOURCE_PATH}{DE_RL_DNN_RES_DIR}network_dnn_%s_{D}_{now_time}.pth'
    pd.DataFrame(columns=['Function', 'Mean', 'Std', 'Convergence']).to_csv(file_path, index=False, mode='w')

    policy_net = PolicyGradient(input_dim=8)

    for func_obj in TRAIN_FUNCS:
        raw_name = func_obj.name
        func_name = raw_name.split(":")[0]
        print(f'######## {func_name} 训练开始 ########')
        func = func_obj(ndim=D)
        start_time = get_now_second()

        # 每 n 次循环记录一次最优解
        min_process = np.zeros((TRAIN_RUNTIME, 1 + int(MAX_CYCLE / RECORD_N)))
        # 多次运行取平均值
        mins = [0.0] * TRAIN_RUNTIME

        for run in range(TRAIN_RUNTIME):
            # 设置随机数种子
            random.seed(SEED * run)
            np.random.seed(SEED * run)
            torch.manual_seed(SEED * run)

            fes = 0
            X, f = initial(func)

            old_best = min(f)
            old_q10 = np.percentile(f, 10)
            old_q25 = np.percentile(f, 25)
            discarded = DiscardedArchive(dim=D, max_cnt=D)

            for it in range(MAX_CYCLE):
                # 获取当前状态
                state = get_state(f, it)

                # 使用策略网络选择F和CR
                F, CR, p, Fw, archF, archP = policy_net.select_action(state)
                if (it + 1) % RECORD_N == 0:
                    print(f'{run + 1}-{it + 1}', end='--')

                V = de_mutation(X, f, F, Fw, p, archF, archP, discarded)
                U = de_crossover(X, V, CR)
                X, f, discarded = de_selection(func, X, f, U, discarded, it)
                f_best = min(f)
                f_q10 = np.percentile(f, 10)
                f_q25 = np.percentile(f, 25)
                fes += NP

                # 计算奖励并缩放
                reward = (old_best - f_best) / abs(old_best) + 0.2 * (old_q10 - f_q10) / abs(old_q10) + 0.1 * (
                        old_q25 - f_q25) / abs(old_q25)
                reward = log_scale(reward)
                reward = reward * (it / 10.0 + 1)
                reward = max(-0.1, min(0.1, reward))
                if (it + 1) % RECORD_N == 0:
                    print(f'reward{reward:.6f}', end=' ## ')

                old_best = f_best
                old_q10 = f_q10
                old_q25 = f_q25

                # 存储经验
                policy_net.store_transition(state, F, CR, p, Fw, archF, archP, reward)
                # 每n次迭代学习一次
                if (it + 1) % DE_PER_RL_UPDATE == 0:
                    policy_net.learn()

                # 记录最优解
                min_v = min(f)
                if it == 0:
                    min_process[run][0] = round(min_v)
                if (it + 1) % RECORD_N == 0:
                    min_process[run][int((it + 1) / RECORD_N)] = round(min_v)

                # 检查是否达到最大函数评估次数
                if fes >= MAX_FES:
                    break

            # 输出当前运行的结果
            print(f'\n{run + 1}.run:  {min_v:.3e}')
            mins[run] = min_v

        # 计算平均值和标准差
        mean, std = np.mean(mins), np.std(mins, ddof=0)
        print(f'FES:  {fes}')
        print(f'Means of {TRAIN_RUNTIME} runs:  {mean:.3e}')
        print(f'Std of {TRAIN_RUNTIME} runs:  {std:.3e}')
        print(f'运行耗时 {get_now_second() - start_time} 秒\n')

        mean_curve = np.mean(min_process, axis=0)
        convergence_str = ';'.join(f'{idx * RECORD_N}:{round(v, 3)}' for idx, v in enumerate(mean_curve))
        # 实时写入
        pd.DataFrame([{'Function': func_name, 'Mean': round(mean, 3), 'Std': round(std, 3),
                       'Convergence': convergence_str}]).to_csv(file_path, index=False, mode='a', header=False)

        # 保存模型参数到本地，每次使用一个函数训练完，就保存一次
        # torch.save(policy_net.policy_net.state_dict(), model_path % func_name)

    torch.save(policy_net.policy.state_dict(), model_path % 'finsh')
    print(f'\n******** 算法运行总耗时 {get_now_second() - main_start_time} 秒 ********\n')
    return now_time


# ---------- 测试 ----------
def de_policy_gradient_test_main(now_time):
    main_start_time = get_now_second()
    file_path = f'{RESOURCE_PATH}{DE_RL_DNN_RES_DIR}de_rl_dnn_test_{D}_{now_time}.csv'
    pd.DataFrame(columns=['Function', 'Mean', 'Std', 'Convergence']).to_csv(file_path, index=False, mode='w')

    # 加载训练好的策略网络模型
    policy_net = PolicyGradient(input_dim=8)
    model_path_full = f'{RESOURCE_PATH}{DE_RL_DNN_RES_DIR}network_dnn_finsh_{D}_{now_time}.pth'
    # 加载训练好的模型
    if os.path.exists(model_path_full):
        policy_net.policy.load_state_dict(torch.load(model_path_full))
    else:
        print(f"模型文件 {model_path_full} 不存在，跳过测试")
        return

    for func_obj in TEST_FUNCS:
        raw_name = func_obj.name
        func_name = raw_name.split(":")[0]
        print(f'######## {func_name} 测试开始 ########')
        func = func_obj(ndim=D)
        start_time = get_now_second()

        # 每 n 次循环记录一次最优解
        min_process = np.zeros((TEST_RUNTIME, 1 + int(MAX_CYCLE / RECORD_N)))
        # 多次运行取平均值
        mins = [0.0] * TEST_RUNTIME

        for run in range(TEST_RUNTIME):
            # 设置随机数种子
            random.seed(SEED * run)
            np.random.seed(SEED * run)
            torch.manual_seed(SEED * run)

            fes = 0
            X, f = initial(func)
            discarded = DiscardedArchive(dim=D, max_cnt=D)
            ls_calls = 0  # 本次运行中 L-BFGS-B 被调用的次数

            for it in range(MAX_CYCLE):
                # 获取当前状态
                state = get_state(f, it)

                # 使用策略网络选择F和CR
                F, CR, p, Fw, archF, archP = policy_net.select_action(state)
                if (it + 1) % RECORD_N == 0:
                    print(f'{run + 1}-{it + 1}', end='--')

                V = de_mutation(X, f, F, Fw, p, archF, archP, discarded)
                U = de_crossover(X, V, CR)
                X, f, discarded = de_selection(func, X, f, U, discarded, it)
                fes += NP

                # ---------- 双通道局部搜索 ----------
                min_v = min(f)  # 保证 min_v 始终有值
                # 只在后 20 % 迭代里做
                if it >= MAX_CYCLE * 0.9:
                    # 1) 全局最优
                    best_idx = np.argmin(f)
                    if np.random.rand() < LS_P:
                        new_x, new_f, _ = refine_best_with_lbfgs(func, X[best_idx])
                        ls_calls += 1
                        if new_f < f[best_idx]:
                            X[best_idx] = new_x
                            f[best_idx] = new_f
                    # 2) 前5%精英（不含最优）随机再挑1个
                    elite_cnt = max(2, int(0.05 * NP))  # ≥2保证切片非空
                    elite_idx = np.argsort(f)[1:elite_cnt]  # 第2名开始
                    if elite_idx.size:
                        pick = np.random.choice(elite_idx)
                        if np.random.rand() < LS_P:
                            new_x, new_f, _ = refine_best_with_lbfgs(func, X[pick])
                            ls_calls += 1
                            if new_f < f[pick]:
                                X[pick] = new_x
                                f[pick] = new_f
                    # 更新当前最优
                    min_v = np.min(f)

                if it == 0:
                    min_process[run][0] = round(min_v)
                if (it + 1) % RECORD_N == 0:
                    min_process[run][int((it + 1) / RECORD_N)] = round(min_v)

                # 检查是否达到最大函数评估次数
                if fes >= MAX_FES:
                    break

            print(f'\n{run + 1}.run:  {min_v:.3e}  (L-BFGS calls: {ls_calls})')
            mins[run] = min_v

        # 计算平均值和标准差
        mean, std = np.mean(mins), np.std(mins, ddof=0)
        print(f'FES:  {fes}')
        print(f'Means of {TEST_RUNTIME} runs:  {mean:.3e}')
        print(f'Std of {TEST_RUNTIME} runs:  {std:.3e}')
        print(f'运行耗时 {get_now_second() - start_time} 秒\n')

        mean_curve = np.mean(min_process, axis=0)
        convergence_str = ';'.join(f'{idx * RECORD_N}:{round(v, 3)}' for idx, v in enumerate(mean_curve))
        # 实时写入
        pd.DataFrame([{'Function': func_name, 'Mean': round(mean, 3), 'Std': round(std, 3),
                       'Convergence': convergence_str}]).to_csv(file_path, index=False, mode='a', header=False)

    print(f'\n******** 算法运行总耗时 {get_now_second() - main_start_time} 秒 ********\n')


# ---------- 主入口 ----------
if __name__ == '__main__':
    # now_time = de_policy_gradient_train_main()
    # de_policy_gradient_test_main(now_time)
    de_policy_gradient_test_main('20250820_1800')
