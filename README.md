# RL-DE: Reinforcement Learning-Controlled Differential Evolution with L-BFGS Refinement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/)

## Overview

This repository contains the implementation of **RL-DE (Reinforcement Learning-controlled Differential Evolution with L-BFGS Refinement)**, a novel optimization framework that integrates reinforcement learning-based parameter control with differential evolution and quasi-Newton local refinement.

**Key Innovation**: RL-DE achieves a paradigm shift from rule-driven to data-driven parameter adaptation through a three-layer collaborative framework:
1. **PG-Tuner**: Policy gradient-based six-dimensional continuous parameter controller
2. **Two-stage mutation switching**: Balances global exploration and local exploitation
3. **LB-Refiner**: L-BFGS-based local refinement module

## Paper Information

**Title**: Reinforcement learning-controlled differential evolution with L-BFGS refinement

**Authors**: Yang Cao, Bingchuan Wu*, Miao Wen

**Affiliation**: 
- School of Computer Science and Engineering, Shenyang Jianzhu University, Shenyang, China
- Liaoning Provincial Key Laboratory of Big Data Management and Analysis of Urban Construction, Shenyang, China
- Shenyang Branch of National Special Computer Engineering Technology Research Center, Shenyang, China

**Corresponding Author**: Bingchuan Wu (183740157@qq.com)

## Features

- **Six-dimensional continuous parameter control**: F, CR, p, Fw, archF, archP via policy gradient
- **Two-stage mutation strategy**: 
  - Early stage (≤50% iterations): DE/rand/2 + Archive for global exploration
  - Late stage (>50% iterations): DE/current-to-pBest-w/1 + Archive for local exploitation
- **External archive mechanism**: Maintains diversity with max-min distance replacement strategy
- **L-BFGS local refinement**: Activated in final 10% iterations with probability 0.2
- **State representation**: 8-dimensional vector capturing population statistics and search progress
- **Reward design**: Multi-objective improvement tracking (best, Q10, Q25)

## Repository Structure

```
├── de_rl_dnn.py              # Main RL-DE algorithm implementation
├── de_rl_dnn_network.py      # PG-Tuner policy network architecture
├── de_base.py                # Baseline DE algorithm for comparison
├── de_reference.py           # Reference algorithms (SaDE, SHADE, ILSHADE, JSO, MPEDE, LSHADE)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── CITATION.cff             # Citation information
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 3.10
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

```
pandas==1.5.3
scikit-learn==1.5.0
matplotlib==3.6.2
opfunu==1.0.4
numpy==1.26.4
torch==1.13.1
scipy==1.15.3
numba==0.61.2
tabulate==0.9.0
```

## Usage

### Training the Policy Network

```python
from de_rl_dnn import de_policy_gradient_train_main

# Train PG-Tuner on CEC2014 benchmark functions
training_timestamp = de_policy_gradient_train_main()
```

### Testing the Trained Model

```python
from de_rl_dnn import de_policy_gradient_test_main

# Test on CEC2017 benchmark functions using trained policy
de_policy_gradient_test_main(training_timestamp)
```

### Running Baseline Algorithms

```bash
# Run baseline DE
python de_base.py

# Run reference algorithms (SaDE, SHADE, ILSHADE, JSO, MPEDE, LSHADE)
python de_reference.py
```

## Algorithm Parameters

| Parameter | Description | Default Value | Range |
|-----------|-------------|---------------|-------|
| NP | Population size | 100 | User-specified |
| D | Problem dimension | 10/30/50/100 | - |
| MAX_FES | Maximum function evaluations | 5000 × D | - |
| SEED | Random seed | 2025 | - |
| TRAIN_RUNTIME | Training runs per function | 5 | - |
| TEST_RUNTIME | Testing runs per function | 30 | - |
| LS_P | L-BFGS activation probability | 0.2 | [0, 1] |
| L-BFGS threshold | Activation iteration threshold | 0.9 | [0, 1] |
| Phase threshold | Strategy switching point | 0.5 | [0, 1] |

## Key Components

### 1. PG-Tuner Policy Network (`de_rl_dnn_network.py`)

**Architecture**:
- Input: 8-dimensional state vector
- Hidden layers: 2 layers × 64 neurons (ReLU activation)
- Output: 6-dimensional Gaussian distribution mean μ

**State Vector (s_t)**:
```
s_t = [f_min, f_Q1, f_med, f_Q3, f_max, f_mean, f_std, log(t/T)]
```

**Action Space (a_t)**:
```
a_t = [F, CR, p, Fw, archF, archP]
```
- F ∈ [0.5, 1.0]: Mutation scaling factor
- CR ∈ [0.4, 1.0]: Crossover probability
- p ∈ [0.05, 0.2]: pBest selection ratio
- Fw ∈ [0.4, 1.0]: Weight scaling factor
- archF ∈ [0.2, 0.4]: Archive disturbance intensity
- archP ∈ [0.05, 0.2]: Archive usage probability

**Reward Function**:
```
r_t = log[(f_best^(t-1) - f_best^(t)) / |f_best^(t-1)|] 
    + 0.2 × log[(f_Q10^(t-1) - f_Q10^(t)) / |f_Q10^(t-1)|]
    + 0.1 × log[(f_Q25^(t-1) - f_Q25^(t)) / |f_Q25^(t-1)|]
```

### 2. Two-Stage Mutation Strategy

**Stage 1 (t ≤ 0.5T) - Global Exploration**:
```
DE/rand/2 + Archive:
v_i = x_r1 + F × (x_r2 - x_r3) + F × (x_r4 - x_r5) 
      + archF × (x_pBest - x_a)  [with probability archP]
```

**Stage 2 (t > 0.5T) - Local Exploitation**:
```
DE/current-to-pBest-w/1 + Archive:
v_i = x_i + Fw × (x_pBest - x_i) + F × (x_r1 - x_r2)
      + archF × (x_pBest' - x_a)  [with probability archP]
```

### 3. External Archive Management

- **Capacity**: |A|_max = D (problem dimension)
- **Storage probability**: p_save(t) = p_min + (p_max - p_min) × γ^t_arch
  - p_max = 1.0, p_min = 0.01, γ_arch = 0.98
- **Overflow handling**: Max-min distance replacement strategy

### 4. LB-Refiner Module

- **Activation condition**: t ≥ 0.9T and rand() < LS_P
- **Max iterations**: 30
- **Convergence threshold**: ε_g = 1e-5
- **Target**: Current best or top 5% elite individuals

## Benchmark Evaluation

### CEC2017 Test Suite

The algorithm is evaluated on 29 functions from the CEC2017 benchmark suite across four dimensions (10D, 30D, 50D, 100D):

| Dimension | Functions Won (out of 27/28) | Friedman Rank |
|-----------|------------------------------|---------------|
| 10D | 11 (40.7%) | 1st (3.17) |
| 30D | 12 (44.4%) | 1st (2.57) |
| 50D | 13 (48.1%) | 1st (2.40) |
| 100D | 14 (50.0%) | 1st (2.88) |

### Comparison Algorithms

- DE (Differential Evolution)
- SaDE (Self-adaptive DE)
- SHADE (Success-History based Adaptive DE)
- ILSHADE (Improved L-SHADE)
- jSO (jSO algorithm)
- MPEDE (Multi-Population Ensemble DE)
- LSHADE (L-SHADE with Linear Population Size Reduction)

## Results Format

Results are saved as CSV files with the following columns:
- `Algorithm`: Algorithm name
- `Function`: Test function name
- `Mean`: Mean best fitness over all runs
- `Std`: Standard deviation of best fitness
- `Convergence`: Semicolon-separated convergence curve (FE:value pairs)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rlde2026,
  title={Reinforcement learning-controlled differential evolution with L-BFGS refinement},
  author={Cao, Yang and Wu, Bingchuan and Wen, Miao},
  journal={PLOS ONE},
  year={2026},
  publisher={Public Library of Science}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was supported by the General Project of Liaoning Provincial Department of Education (LJ212410153017)
- The CEC2014/2017 benchmark functions are provided by the opfunu library
- Built with PyTorch for deep learning components

## Contact

For questions or issues, please contact:
- Bingchuan Wu: 183740157@qq.com
- Open an issue on the GitHub repository