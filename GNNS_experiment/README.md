# 图神经网络假新闻检测项目

## 项目简介

本项目实现了基于图神经网络的假新闻检测系统，使用多种GNN模型对新闻文本进行图结构化建模，实现假新闻与真实新闻的分类任务。

## 项目特点

- **多种GNN模型支持**：实现了GCN、GAT、GraphSAGE、GIN和TEXT-GCN五种主流图神经网络模型
- **少样本学习**：采用分层采样策略，每个类别仅使用20个训练样本
- **文本图构建**：将文本转换为图结构，句子作为节点，相似度作为边权重
- **详细评估**：提供准确率、精确率、召回率、F1分数等多种评估指标

## 项目结构

```
图神经网络-Few/
├── data/                          # 数据目录
│   ├── politifact/               # Politifact数据集
│   │   ├── train.jsonl          # 训练集
│   │   ├── test.jsonl           # 测试集
│   │   └── val.jsonl            # 验证集
│   ├── gossipcop/               # GossipCop数据集
│   └── CoAID-master/            # CoAID数据集
├── train_GCN.py                  # GCN模型训练脚本
├── train_GAT.py                  # GAT模型训练脚本
├── train_GraphSAGE.py            # GraphSAGE模型训练脚本
├── train_GIN.py                  # GIN模型训练脚本
├── train_TEXT_GCN.py             # TEXT-GCN模型训练脚本
├── README.md                     # 项目说明文档
└── requirements.txt              # 依赖包列表
```

## 实验环境

- **操作系统**: Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-139-generic x86_64)
- **Python版本**: Python 3.11.13
- **GPU**: NVIDIA A100 80GB PCIe (CUDA Version: 13.0, Driver Version: 580.76.05)
- **网络**: 稳定的互联网连接（用于API调用和模型下载）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

确保数据文件位于 `data/` 目录下，支持JSONL格式：

```json
{"text": "新闻文本内容", "label": "fake"}
{"text": "真实新闻内容", "label": "real"}
```

### 2. 运行模型

#### GCN模型
```bash
python train_GCN.py
```

#### GAT模型
```bash
python train_GAT.py
```

#### GraphSAGE模型
```bash
python train_GraphSAGE.py
```

#### GIN模型
```bash
python train_GIN.py
```

#### TEXT-GCN模型
```bash
python train_TEXT_GCN.py
```

## 模型介绍

### 1. GCN (Graph Convolutional Network)
- 基础图卷积网络
- 简单高效，适合图分类任务
- 支持多层传播和dropout

### 2. GAT (Graph Attention Network)
- 图注意力网络
- 多头注意力机制
- 自动学习节点间的重要性权重

### 3. GraphSAGE (Graph Sample and Aggregate)
- 归纳式图神经网络
- 支持多种聚合函数（mean, max, add）
- 适合大规模图数据

### 4. GIN (Graph Isomorphism Network)
- 图同构网络
- 强大的图表示学习能力
- 包含MLP模块和可训练epsilon参数

### 5. TEXT-GCN
- 专门用于文本的图卷积网络
- 构建词-文档异构图
- 使用PMI计算词间关系


## 数据配置
```python
TRAIN_FILE = './data/politifact/train.jsonl'  # 训练数据路径
TEST_FILE = './data/politifact/test.jsonl'    # 测试数据路径
```

## 少样本学习设置

项目采用分层采样策略：
- 每个类别（fake/real）随机选择20个训练样本
- 测试集保持不变
- 设置随机种子确保结果可复现

## 评估指标

模型训练完成后会输出以下指标：
- **准确率 (Accuracy)**：正确预测的比例
- **精确率 (Precision)**：预测为正例中实际为正例的比例
- **召回率 (Recall)**：实际正例中被正确预测的比例
- **F1分数**：精确率和召回率的调和平均数

## 自定义数据

如需使用自定义数据集，请确保：

1. 数据格式为JSONL
2. 每行包含文本内容和标签
3. 支持的字段名：`text`, `message`, `content`, `title`
4. 标签值为：`fake` 或 `real`

示例：
```json
{"text": "这是一条假新闻", "label": "fake"}
{"title": "真实新闻标题", "label": "real"}
```
