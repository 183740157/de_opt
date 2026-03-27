#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：de_opt
@File    ：de_rl_dnn_network.py
@IDE     ：PyCharm
@Author  ：wubc
@Date    ：2025/04/27 16:10
描述      ：策略梯度网络——全连接神经网络
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from constants.algo_const import SEED

# ------------------------- 随机种子 -------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------- 网络结构 -------------------------
class PolicyNetwork(nn.Module):
    """
    输出两个连续动作 F 和 CR 的均值 μ 和对数标准差 logσ
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, action_std_init: float = 0.3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 两层带残差的 MLP
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, 6)
        self.register_buffer('log_std', torch.ones(6) * np.log(action_std_init))

    def forward(self, x):
        z = torch.relu(self.input_proj(x))      # [B, hidden_dim]

        # 残差块 1
        out = torch.relu(self.fc1(z))
        z = out + z                             # skip connection

        # 残差块 2
        out = torch.relu(self.fc2(z))
        z = out + z                             # skip connection

        raw = self.mu_head(z)

        tau = 8.0
        mu_F = torch.sigmoid(raw[:, 0] / tau) * 0.5 + 0.5  # F ∈ [0.5, 1.0]
        mu_CR = torch.sigmoid(raw[:, 1] / tau) * 0.6 + 0.4  # CR ∈ [0.4, 1.0]
        mu_p = torch.sigmoid(raw[:, 2] / tau) * 0.15 + 0.05  # p ∈ [0.05, 0.2]
        mu_Fw = torch.sigmoid(raw[:, 3] / tau) * 0.6 + 0.4  # Fw ∈ [0.4, 1.0]
        mu_archF = torch.sigmoid(raw[:, 4] / tau) * 0.2 + 0.2  # F_a ∈ [0.2, 0.4]
        mu_archP = torch.sigmoid(raw[:, 5] / tau) * 0.15 + 0.05  # P_a ∈ [0.05, 0.2]
        mu = torch.stack([mu_F, mu_CR, mu_p, mu_Fw, mu_archF, mu_archP], dim=1)
        log_std = self.log_std.expand_as(mu)
        return mu, log_std


# ------------------------- 策略封装 -------------------------
class PolicyGradient:
    def __init__(self, input_dim: int, lr: float = 1e-3, gamma: float = 0.9, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma

        self.policy = PolicyNetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = []  # [(state, action, reward)]

    # ------------- 动作采样 -------------
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, log_std = self.policy(state)
            # 训练阶段：采样
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            F = torch.clamp(action[0, 0], 0.5, 1.0)
            CR = torch.clamp(action[0, 1], 0.4, 1.0)
            p = torch.clamp(action[0, 2], 0.05, 0.2)
            Fw = torch.clamp(action[0, 3], 0.4, 1.0)
            archF = torch.clamp(action[0, 4], 0.2, 0.4)
            archP = torch.clamp(action[0, 5], 0.05, 0.2)
            return F.item(), CR.item(), p.item(), Fw.item(), archF.item(), archP.item()

    # ------------- 经验存储 -------------
    def store_transition(self, state, F, CR, p, Fw, archF, archP, reward):
        action = torch.FloatTensor([F, CR, p, Fw, archF, archP])
        self.memory.append((state, action, reward))

    # ------------- 训练 -------------
    def learn(self):
        if len(self.memory) == 0:
            return

        # 拆包
        states = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)  # [B, input_dim]
        actions = torch.stack([t[1] for t in self.memory]).to(self.device)  # [B, 2]
        rewards = torch.FloatTensor([t[2] for t in self.memory]).to(self.device)  # [B]

        # 计算折扣回报
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 重算分布
        mu, log_std = self.policy(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)  # [B]

        # REINFORCE 损失
        loss = -(log_prob * returns).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        # self.print_param()
        self.optimizer.step()

        # 清空记忆
        self.memory.clear()

    def print_param(self):
        '''查看神经网络梯度'''
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                print(name, param.grad.norm().item())
