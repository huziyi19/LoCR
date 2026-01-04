# file: train_GraphSAGE.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk
import jsonlines
from sentence_transformers import SentenceTransformer, util
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- 全局配置 ---
# 数据路径
TRAIN_FILE = './data/politifact/train.jsonl'
TEST_FILE = './data/politifact/test.jsonl'

# 模型参数
EMBEDDING_DIM = 384  # 使用的句子嵌入模型 all-MiniLM-L6-v2 的维度
HIDDEN_DIM = 128     # GraphSAGE隐藏层维度
NUM_CLASSES = 2      # 类别数: fake, real

# 训练超参数
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
EPOCHS = 10
BATCH_SIZE = 32

# GraphSAGE特定参数
NUM_LAYERS = 2       # GraphSAGE层数
AGG = 'mean'         # 聚合方式: mean, max, add

# 其他配置
SIMILARITY_THRESHOLD = 0.5 # 构建图的相似度阈值

# --- 准备工作 ---
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"将使用的设备: {device}")

# 1. 下载NLTK的句子分割器模型 (仅需首次运行)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("正在下载NLTK句子分割器模型 'punkt'...")
    nltk.download('punkt')

# 2. 加载预训练的句子嵌入模型 (一次性加载，避免重复)
print("正在加载句子嵌入模型 (all-MiniLM-L6-v2)...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("模型加载完成。")


# --- 数据处理函数 ---
def load_data_from_jsonl(file_path):
    """从jsonl文件中加载文本和标签"""
    texts, labels = [], []
    label_map = {'fake': 0, 'real': 1}
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            # 支持多种文本字段名：text, message, content等
            text_field = None
            for field in ['text', 'message', 'content', 'title']:
                if field in obj and obj[field]:
                    text_field = obj[field]
                    break

            if text_field and 'label' in obj:
                texts.append(text_field)
                labels.append(label_map[obj['label']])
    return texts, labels

def build_graph_from_text(text: str) -> Data:
    """将单篇新闻文本转换为图数据对象"""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return None # 句子太少无法建图

    # 使用已加载的模型生成嵌入
    with torch.no_grad():
        node_features = sentence_model.encode(sentences, convert_to_tensor=True, device=device)

    similarity_matrix = util.pytorch_cos_sim(node_features, node_features)

    edge_list = []
    num_nodes = len(sentences)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = similarity_matrix[i, j].item()
            if similarity > SIMILARITY_THRESHOLD:
                edge_list.append([i, j])
                edge_list.append([j, i]) # 无向图

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data

def preprocess_data(texts, labels):
    """将整个数据集转换为图对象列表"""
    graph_list = []
    print("正在将文本数据转换为图结构...")
    for text, label in tqdm(zip(texts, labels), total=len(texts)):
        graph = build_graph_from_text(text)
        if graph is not None:
            graph.y = torch.tensor([label], dtype=torch.long)
            graph_list.append(graph)
    return graph_list


# --- 模型定义 ---
class GraphSAGEClassifier(nn.Module):
    """GraphSAGE图分类模型"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, agg='mean'):
        super(GraphSAGEClassifier, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        # 第一层
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=agg))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=agg))

        # 最后一层
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=agg))

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 通过所有GraphSAGE层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 最后一层不激活
                x = F.relu(x)
                x = self.dropout(x)

        # 图级别池化
        graph_embedding = global_mean_pool(x, batch)

        # 分类
        output = self.classifier(graph_embedding)
        return output


# --- 训练与评估循环 ---
def train_loop(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="训练中"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def eval_loop(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(loader, desc="评估中"):
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    return accuracy, precision, recall, f1


# --- 主执行函数 ---
if __name__ == "__main__":
    # 1. 加载数据
    print("="*50)
    print("GraphSAGE图神经网络训练")
    print("="*50)
    train_texts, train_labels = load_data_from_jsonl(TRAIN_FILE)
    test_texts, test_labels = load_data_from_jsonl(TEST_FILE)

    # 分层采样：从real样本中抽取20个，从fake样本中抽取20个
    import random
    random.seed(42)  # 设置随机种子确保结果可复现

    # 按类别分组
    real_indices = [i for i, label in enumerate(train_labels) if label == 1]  # real=1
    fake_indices = [i for i, label in enumerate(train_labels) if label == 0]  # fake=0

    print(f"原始训练数据分布: real={len(real_indices)}, fake={len(fake_indices)}")

    # 从每个类别中抽取20个样本
    sample_size_per_class = 20

    # 如果样本不足20个，则取全部
    real_sample_size = min(sample_size_per_class, len(real_indices))
    fake_sample_size = min(sample_size_per_class, len(fake_indices))

    # 随机采样
    random.shuffle(real_indices)
    random.shuffle(fake_indices)

    sampled_real_indices = real_indices[:real_sample_size]
    sampled_fake_indices = fake_indices[:fake_sample_size]

    # 合并样本
    sampled_indices = sampled_real_indices + sampled_fake_indices
    random.shuffle(sampled_indices)  # 打乱顺序

    # 构建新的训练数据
    train_texts = [train_texts[i] for i in sampled_indices]
    train_labels = [train_labels[i] for i in sampled_indices]

    # 统计最终样本分布
    final_real_count = sum(1 for label in train_labels if label == 1)
    final_fake_count = sum(1 for label in train_labels if label == 0)

    print(f"分层采样完成: real={final_real_count}, fake={final_fake_count}, 总计={len(train_texts)} 条训练样本")

    # 2. 预处理数据为图格式
    train_graphs = preprocess_data(train_texts, train_labels)
    test_graphs = preprocess_data(test_texts, test_labels)
    print(f"图构建完成: {len(train_graphs)} 个有效训练图, {len(test_graphs)} 个有效测试图。")

    # 3. 创建DataLoader
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化GraphSAGE模型、优化器和损失函数
    model = GraphSAGEClassifier(
        input_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        agg=AGG
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print("\n模型、优化器、损失函数已初始化。")
    print(f"GraphSAGE配置: {NUM_LAYERS}层, 聚合方式={AGG}")
    print(model)

    # 5. 开始训练和评估
    print("\n" + "="*50)
    print("开始训练...")

    # 用于记录所有epoch的结果
    all_train_losses = []
    all_test_accuracies = []
    all_test_precisions = []
    all_test_recalls = []
    all_test_f1s = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_loop(model, train_loader, optimizer, criterion)

        # 在每个epoch后进行评估
        test_acc, test_prec, test_rec, test_f1 = eval_loop(model, test_loader)

        # 记录结果
        all_train_losses.append(train_loss)
        all_test_accuracies.append(test_acc)
        all_test_precisions.append(test_prec)
        all_test_recalls.append(test_rec)
        all_test_f1s.append(test_f1)

        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        print(f"训练损失 (Train Loss): {train_loss:.4f}")
        print(f"测试集准确率 (Test Acc): {test_acc:.4f}")
        print(f"测试集 F1分数 (Test F1): {test_f1:.4f}")
        print(f"测试集 精确率 (Test Precision): {test_prec:.4f}")
        print(f"测试集 召回率 (Test Recall): {test_rec:.4f}")
        print("-"*50)

    # 计算平均值
    avg_train_loss = sum(all_train_losses) / len(all_train_losses)
    avg_test_acc = sum(all_test_accuracies) / len(all_test_accuracies)
    avg_test_prec = sum(all_test_precisions) / len(all_test_precisions)
    avg_test_rec = sum(all_test_recalls) / len(all_test_recalls)
    avg_test_f1 = sum(all_test_f1s) / len(all_test_f1s)

    # 找出最佳结果
    best_f1_idx = max(range(len(all_test_f1s)), key=lambda i: all_test_f1s[i])
    best_acc_idx = max(range(len(all_test_accuracies)), key=lambda i: all_test_accuracies[i])

    print("\n" + "="*50)
    print("GraphSAGE训练完成！")
    print("="*50)
    print("训练结果统计:")
    print(f"平均训练损失: {avg_train_loss:.4f}")
    print(f"平均测试准确率: {avg_test_acc:.4f}")
    print(f"平均测试精确率: {avg_test_prec:.4f}")
    print(f"平均测试召回率: {avg_test_rec:.4f}")
    print(f"平均测试F1分数: {avg_test_f1:.4f}")
    print("\n最佳结果:")
    print(f"最高F1分数: {all_test_f1s[best_f1_idx]:.4f} (第{best_f1_idx+1}轮)")
    print(f"最高准确率: {all_test_accuracies[best_acc_idx]:.4f} (第{best_acc_idx+1}轮)")
    print("="*50)