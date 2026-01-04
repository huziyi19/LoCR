## Project Overview
This project implements a fake news detection system based on Graph Neural Networks (GNNs). It leverages multiple GNN models to perform graph-structured modeling on news text, enabling the classification task of distinguishing fake news from real news.

## Project Features
- **Multi-GNN Model Support**: Implements five mainstream graph neural network models: GCN, GAT, GraphSAGE, GIN, and TEXT-GCN.
- **Few-Shot Learning**: Adopts a stratified sampling strategy with only 20 training samples per category.
- **Text Graph Construction**: Converts text into graph structures where sentences serve as nodes and similarity scores act as edge weights.
- **Comprehensive Evaluation**: Provides multiple evaluation metrics including Accuracy, Precision, Recall, and F1-Score.

## Project Structure
```
Graph-Neural-Network-Few/
├── data/                          # Data directory
│   ├── politifact/               # Politifact dataset
│   │   ├── train.jsonl          # Training set
│   │   ├── test.jsonl           # Test set
│   │   └── val.jsonl            # Validation set
│   ├── gossipcop/               # GossipCop dataset
│   └── CoAID-master/            # CoAID dataset
├── train_GCN.py                  # GCN model training script
├── train_GAT.py                  # GAT model training script
├── train_GraphSAGE.py            # GraphSAGE model training script
├── train_GIN.py                  # GIN model training script
├── train_TEXT_GCN.py             # TEXT-GCN model training script
├── README.md                     # Project documentation
└── requirements.txt              # Dependencies list
```

## Experimental Environment
- **Operating System**: Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-139-generic x86_64)
- **Python Version**: Python 3.11.13
- **GPU**: NVIDIA A100 80GB PCIe (CUDA Version: 13.0, Driver Version: 580.76.05)
- **Network**: Stable internet connection (required for API calls and model downloads)

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start
### 1. Data Preparation
Ensure data files are placed in the `data/` directory, supporting JSONL format:
```json
{"text": "News text content", "label": "fake"}
{"text": "Real news content", "label": "real"}
```

### 2. Run the Models
#### GCN Model
```bash
python train_GCN.py
```

#### GAT Model
```bash
python train_GAT.py
```

#### GraphSAGE Model
```bash
python train_GraphSAGE.py
```

#### GIN Model
```bash
python train_GIN.py
```

#### TEXT-GCN Model
```bash
python train_TEXT_GCN.py
```

## Model Introduction
### 1. GCN (Graph Convolutional Network)
- Basic graph convolutional network
- Simple and efficient, suitable for graph classification tasks
- Supports multi-layer propagation and dropout

### 2. GAT (Graph Attention Network)
- Graph attention network
- Multi-head attention mechanism
- Automatically learns importance weights between nodes

### 3. GraphSAGE (Graph Sample and Aggregate)
- Inductive graph neural network
- Supports multiple aggregation functions (mean, max, add)
- Suitable for large-scale graph data

### 4. GIN (Graph Isomorphism Network)
- Graph isomorphism network
- Strong graph representation learning capability
- Includes MLP module and trainable epsilon parameter

### 5. TEXT-GCN
- Graph convolutional network specially designed for text
- Constructs word-document heterogeneous graphs
- Uses PMI (Pointwise Mutual Information) to calculate word relationships

## Data Configuration
```python
TRAIN_FILE = './data/politifact/train.jsonl'  # Training data path
TEST_FILE = './data/politifact/test.jsonl'    # Test data path
```

## Few-Shot Learning Settings
The project adopts a stratified sampling strategy:
- Randomly select 20 training samples for each category (fake/real)
- Keep the test set unchanged
- Set random seeds to ensure reproducibility of results

## Evaluation Metrics
After model training, the following metrics will be output:
| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correctly predicted samples |
| **Precision** | Proportion of true positives among predicted positives |
| **Recall** | Proportion of true positives correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall |

## Custom Data
To use a custom dataset, ensure:
1. Data format is JSONL
2. Each line contains text content and a label
3. Supported field names: `text`, `message`, `content`, `title`
4. Label values: `fake` or `real`

Example:
```json
{"text": "This is a fake news article", "label": "fake"}
{"title": "Real news title", "label": "real"}
```
