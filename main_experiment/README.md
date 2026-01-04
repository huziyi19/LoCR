# LFND: 大模型驱动的新闻假新闻检测系统

## 项目简介

本项目实现了一个基于大语言模型的新闻假新闻检测系统，结合了逻辑推理和图神经网络技术。该系统通过以下方法对新闻进行真假检测：

1. **方法一**: 大模型综合判断 (LLM-Only)
2. **方法二**: 完整LFND系统 (大模型 + 逻辑推理 + 图神经网络 + 迭代修正)

## 核心技术

- **自然语言处理**: 使用大模型进行句子真实性评估
- **图神经网络**: 基于句子相似度构建图结构，使用GCN进行特征传播
- **逻辑推理**: 利用自然语言推理(NLI)模型检测句子间的逻辑关系
- **迭代修正**: 检测置信度变化并自动修正不一致的判断

## 支持的大模型

本系统支持以下大语言模型（通过OpenRouter API）：

- **deepseek/deepseek-chat** - DeepSeek聊天模型（默认推荐）
- **openai/gpt-4o-mini** - OpenAI GPT-4o Mini
- **google/gemini-2.5-flash** - Google Gemini 2.5 Flash
- **qwen/qwen3-32b** - 通义千问3 32B模型
- **meta-llama/llama-3.3-70b-instruct** - Meta Llama 3.3 70B指令模型

您可以在 `.env` 文件中配置使用的模型。

## 项目结构

```
├── data/                   # 数据文件夹
│   ├── gossipcop/         # GossipCop数据集
│   └── politifact/        # PolitiFact数据集
├── LFND.py                # 主程序文件
├── load_data.py           # 数据加载和预处理
├── split_and_detect.py    # 句子分割和置信度检测
├── text_to_graph.py       # 句子相似度图构建
├── Logic_propagator.py    # 逻辑推理模块
├── GCN_propagator.py      # 图卷积网络模块
├── .env.example           # 环境变量模板
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明文档
```

## 安装说明

### 实验环境

- **操作系统**: Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-139-generic x86_64)
- **Python版本**: Python 3.11.13
- **GPU**: NVIDIA A100 80GB PCIe (CUDA Version: 13.0, Driver Version: 580.76.05)
- **网络**: 稳定的互联网连接（用于API调用和模型下载）

### 安装步骤

1. 克隆项目到本地
2. 创建并激活虚拟环境
3. 安装依赖包
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥和模型选择
```

## 使用说明

### 数据准备

1. 将数据集放入 `data/` 目录下
2. 支持的格式：
   - `.jsonl` 格式 (如 GossipCop, PolitiFact)
   - `.csv` 格式

数据格式要求：
- `text`: 新闻正文内容
- `label`: 真实标签 (`real` 或 `fake`)
- `id`: 可选，新闻唯一标识符
- `title`: 可选，新闻标题

### 模型配置

在 `.env` 文件中配置您的大模型：

```bash
# --- OpenRouter & LLM Configuration ---
OPENROUTER_API_KEY=""
OPENROUTER_MODEL="qwen/qwen3-32b"
OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions"
```

可选模型列表：
- deepseek/deepseek-chat
- openai/gpt-4o-mini
- google/gemini-2.5-flash
- qwen/qwen3-32b
- meta-llama/llama-3.3-70b-instruct

您可以直接修改 `OPENROUTER_MODEL` 字段来切换不同的模型。

### 运行实验

运行主程序进行批量检测：
```bash
python LFND.py
```

程序将自动：
1. 加载数据集
2. 对每篇新闻执行检测方法
3. 输出详细的比较结果
4. 生成最终的准确率报告

### 输出结果

程序会输出每个样本的详细检测结果，最终会输出各方法的准确率对比。

## 核心算法说明

### 1. 句子分割与置信度评估
- 使用大模型将新闻文本分割为语义完整的句子
- 对每个句子进行0-1区间的真实性置信度评估

### 2. 图构建
- 基于句子嵌入的余弦相似度构建图结构
- 边权重表示句子间的语义相似度

### 3. 逻辑推理
- 使用NLI模型检测句子间的蕴含、矛盾关系
- 实现"否定"和"蕴含"两种逻辑关系的传播

### 4. 图卷积传播
- 使用GCN在相似度图上进行置信度传播
- 填补缺失的置信度值

### 5. 迭代修正
- 检测置信度变化超过阈值的节点
- 调用大模型重新评估这些句子
- 迭代直到收敛或达到最大轮数


## 依赖说明

主要依赖包：
- `openai`: OpenAI API客户端
- `torch`: PyTorch深度学习框架
- `torch-geometric`: 图神经网络库
- `transformers`: Hugging Face transformers
- `sentence-transformers`: 句子嵌入模型
- `networkx`: 图处理库
- `numpy`: 数值计算
- `python-dotenv`: 环境变量管理

## 注意事项

1. **API配额**: 使用OpenRouter API需要有效的API密钥和足够的配额
2. **模型加载**: 首次运行时会自动下载预训练模型，需要稳定的网络连接
3. **内存使用**: 大规模数据处理时注意内存使用情况
4. **GPU加速**: 建议使用GPU以获得更快的推理速度

