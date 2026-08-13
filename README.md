[Uploading README.md…]()
# 向量计算通用仿真框架
Vector General Simulation Framework（VecGPU-Sim）

## 项目概述
本项目是面向复杂网络动力学仿真的向量计算通用仿真框架，基于向量化运算实现高效动力学模拟，内置三种示例。

## 目录结构

├── spread_framework_new_sir.py        # SIR
├── spread_framework_new_siirr.py        # SIIRR
├── spread_framework_new_seior.py        # SEIOR
└── network/          # 部分网络拓扑数据集存放目录

## 说明
1. `spread_framework_new_sir.py`、`spread_framework_new_siirr.py`、`spread_framework_new_seior.py` 分别对应三种不同动力学模型，共享底层向量计算仿真框架；
2. 所有网络拓扑文件统一放置于 `network` 文件夹；
3. 由于完整网络数据集文件体积过大，仓库内仅存放**部分网络样本文件**，如需完整数据集，请联系，后续补充到 `network` 目录。

## 环境依赖
pip install -r requirements.txt
