# file: train_TEXT_GCN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk
import jsonlines
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import networkx as nx

# --- 全局配置 ---
# 数据路径
TRAIN_FILE = './data/politifact/train.jsonl'
TEST_FILE = './data/politifact/test.jsonl'

# 模型参数
EMBEDDING_DIM = 300     # 词向量维度 (增加特征表达)
HIDDEN_DIM = 256        # GCN隐藏层维度 (增加模型容量)
NUM_CLASSES = 2         # 类别数: fake, real

# 训练超参数
LEARNING_RATE = 5e-4    # 降低学习率
WEIGHT_DECAY = 1e-3     # 增加正则化
EPOCHS = 30             # 增加训练轮数
BATCH_SIZE = 64

# TEXT-GCN特定参数
MAX_VOCAB_SIZE = 3000   # 最大词汇表大小 (减少噪声词)
MIN_WORD_FREQ = 3       # 最小词频 (提高阈值)
WINDOW_SIZE = 10        # 滑动窗口大小 (减少无关连接)
PMI_THRESHOLD = 0.1     # PMI阈值 (过滤弱连接)
WORD_CO_OCCUR_THRESHOLD = 3  # 词共现阈值 (提高要求)
TFIDF_MAX_FEATURES = 3000    # TF-IDF最大特征数

# --- 准备工作 ---
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"将使用的设备: {device}")

# 1. 下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("正在下载NLTK句子分割器模型 'punkt'...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("正在下载NLTK停用词...")
    try:
        nltk.download('stopwords')
    except:
        print("无法下载停用词，将跳过停用词过滤...")

# --- 数据加载函数 ---
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

# --- TEXT-GCN核心组件 ---
class Vocabulary:
    """词汇表管理"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        self.idx2word[0] = '<PAD>'
        self.idx2word[1] = '<UNK>'
        self.idx = 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        self.word_freq[word] += 1

    def __len__(self):
        return len(self.word2idx)

class TextGraphBuilder:
    """构建文本图的核心类"""
    def __init__(self, window_size=20, min_freq=2, pmi_threshold=0.0):
        self.window_size = window_size
        self.min_freq = min_freq
        self.pmi_threshold = pmi_threshold
        self.vocab = Vocabulary()
        self.word_co_occurrence = defaultdict(Counter)
        self.word_count = Counter()
        self.total_words = 0
        self.saved_edges = []
        self.saved_edge_weights = []

    def build_vocabulary(self, texts):
        """构建词汇表"""
        print("构建词汇表...")
        all_words = []

        for text in tqdm(texts, desc="处理文本"):
            # 分词和预处理
            words = self._preprocess_text(text)
            all_words.extend(words)

        # 统计词频
        word_freq = Counter(all_words)

        # 过滤低频词
        filtered_words = [word for word in all_words
                         if word_freq[word] >= self.min_freq][:MAX_VOCAB_SIZE]

        # 构建词汇表
        for word in filtered_words:
            self.vocab.add_word(word)

        print(f"词汇表大小: {len(self.vocab)}")
        return self.vocab

    def _preprocess_text(self, text):
        """文本预处理"""
        # 基本清理
        text = text.lower()
        # 分词
        words = nltk.word_tokenize(text)
        # 过滤标点符号和短词
        words = [word for word in words if word.isalpha() and len(word) > 1]
        return words

    def calculate_co_occurrence(self, texts):
        """计算词共现"""
        print("计算词共现...")
        self.word_count = Counter()
        self.total_words = 0

        for text in tqdm(texts, desc="计算共现"):
            words = self._preprocess_text(text)
            # 过滤词汇表中的词
            words = [word for word in words if word in self.vocab.word2idx]

            # 滑动窗口统计共现
            for i in range(len(words)):
                self.word_count[words[i]] += 1
                self.total_words += 1

                # 窗口内的词
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)

                for j in range(start, end):
                    if i != j:
                        self.word_co_occurrence[words[i]][words[j]] += 1

    def calculate_pmi(self):
        """计算点间互信息PMI"""
        print("计算PMI...")
        edges = []
        edge_weights = []

        word_ids = {word: idx for word, idx in self.vocab.word2idx.items()
                   if word not in ['<PAD>', '<UNK>']}

        for word1, co_occur_dict in self.word_co_occurrence.items():
            if word1 not in word_ids:
                continue

            for word2, co_occurrence in co_occur_dict.items():
                if word2 not in word_ids or word1 == word2:
                    continue

                # 计算PMI
                pmi = np.log(
                    (co_occurrence * self.total_words) /
                    (self.word_count[word1] * self.word_count[word2])
                )

                if pmi > self.pmi_threshold and co_occurrence >= WORD_CO_OCCUR_THRESHOLD:
                    idx1, idx2 = word_ids[word1], word_ids[word2]
                    edges.append([idx1, idx2])
                    edge_weights.append(pmi)

        return edges, edge_weights

    def build_word_feature_matrix(self, texts):
        """构建词特征矩阵（使用TF-IDF）"""
        print("构建词特征矩阵...")

        # 准备语料
        corpus = []
        for text in texts:
            words = self._preprocess_text(text)
            words = [word for word in words if word in self.vocab.word2idx]
            corpus.append(' '.join(words))

        # 计算TF-IDF
        vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, vocabulary=self.vocab.word2idx, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # 转换为词特征矩阵
        word_features = np.zeros((len(self.vocab), EMBEDDING_DIM))

        # 将TF-IDF值分配给词节点
        for word, idx in self.vocab.word2idx.items():
            if word in vectorizer.vocabulary_ and word not in ['<PAD>', '<UNK>']:
                # 获取该词在所有文档中的TF-IDF值
                word_tfidf = tfidf_matrix[:, vectorizer.vocabulary_[word]].toarray().flatten()

                # 如果文档数量少于EMBEDDING_DIM，用零填充
                if len(word_tfidf) < EMBEDDING_DIM:
                    padded_features = np.zeros(EMBEDDING_DIM)
                    padded_features[:len(word_tfidf)] = word_tfidf
                    word_features[idx] = padded_features
                else:
                    word_features[idx] = word_tfidf[:EMBEDDING_DIM]

        # 对于没有TF-IDF值的词，使用小的随机值初始化
        for i in range(len(self.vocab)):
            if np.sum(np.abs(word_features[i])) < 1e-6:  # 如果全为零
                word_features[i] = np.random.normal(0, 0.1, EMBEDDING_DIM)

        # 归一化
        word_features = normalize(word_features, norm='l2')
        return word_features

    def build_text_graph(self, texts, labels, use_existing_vocab=False):
        """构建完整的文本图"""
        print("构建TEXT-GCN图...")

        # 1. 构建词汇表（如果需要）
        if not use_existing_vocab and len(self.vocab.word2idx) <= 2:
            self.build_vocabulary(texts)

        # 2. 计算共现（仅在训练时）
        if not use_existing_vocab:
            self.calculate_co_occurrence(texts)

        # 3. 计算PMI并构建边（仅在训练时）
        if not use_existing_vocab:
            edges, edge_weights = self.calculate_pmi()
            self.saved_edges = edges
            self.saved_edge_weights = edge_weights
            print(f"构建了 {len(edges)} 条边")
        else:
            # 测试时使用保存的词-词边
            edges, edge_weights = self.saved_edges, self.saved_edge_weights

        # 4. 构建特征矩阵
        word_features = self.build_word_feature_matrix(texts)

        # 5. 添加文档节点和边
        doc_features = []
        doc_edges = []
        doc_edge_weights = []

        for doc_id, (text, label) in enumerate(zip(texts, labels)):
            words = self._preprocess_text(text)
            words = [word for word in words if word in self.vocab.word2idx]

            # 文档节点特征（词向量的平均）
            if words:
                doc_feature = np.mean([word_features[self.vocab.word2idx[word]] for word in words], axis=0)
            else:
                doc_feature = np.random.normal(0, 0.1, EMBEDDING_DIM)

            doc_features.append(doc_feature)

            # 连接文档和词节点
            word_set = set(words)
            for word in word_set:
                word_idx = self.vocab.word2idx[word]
                doc_idx = len(self.vocab) + doc_id

                # 文档->词边
                doc_edges.append([doc_idx, word_idx])
                doc_edge_weights.append(1.0)

        # 6. 合并所有节点和边
        all_features = np.vstack([word_features, doc_features])
        all_edges = edges + doc_edges
        all_weights = edge_weights + doc_edge_weights

        # 7. 构建PyTorch Geometric数据对象
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(all_weights, dtype=torch.float)

        x = torch.tensor(all_features, dtype=torch.float)

        # 为词节点设置虚拟标签(-1)，只保留文档节点的真实标签
        y_with_word_labels = torch.tensor([-1] * len(self.vocab) + labels, dtype=torch.long)

        # 节点掩码（区分词节点和文档节点）
        num_word_nodes = len(self.vocab)
        doc_mask = torch.zeros(len(all_features), dtype=torch.bool)
        doc_mask[num_word_nodes:] = True

        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y_with_word_labels,
                   doc_mask=doc_mask, num_word_nodes=num_word_nodes)


# --- 模型定义 ---
class TEXTGCN(nn.Module):
    """TEXT-GCN模型"""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(TEXTGCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # 增加第三层
        self.dropout = nn.Dropout(dropout)

        # 增加一个中间层
        self.pre_classifier = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, doc_mask = data.x, data.edge_index, data.edge_attr, data.doc_mask

        # 三层GCN
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # 只对文档节点进行分类
        doc_embeddings = x[doc_mask]
        doc_embeddings = F.relu(self.pre_classifier(doc_embeddings))
        output = self.classifier(doc_embeddings)

        return output


# --- 训练与评估循环 ---
def train_loop(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # 只对文档节点计算损失
    doc_output = out  # 模型已经只输出文档节点
    # 获取文档节点的真实标签（过滤掉词节点的虚拟标签）
    doc_labels = data.y[data.doc_mask]

    loss = criterion(doc_output, doc_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_loop(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)

    # 只对文档节点进行评估
    doc_output = out  # 模型已经只输出文档节点
    # 获取文档节点的真实标签（过滤掉词节点的虚拟标签）
    doc_labels = data.y[data.doc_mask]

    pred = doc_output.argmax(dim=1)
    true = doc_labels

    accuracy = accuracy_score(true.cpu(), pred.cpu())
    precision, recall, f1, _ = precision_recall_fscore_support(true.cpu(), pred.cpu(), average='binary')
    return accuracy, precision, recall, f1


# --- 主执行函数 ---
if __name__ == "__main__":
    # 1. 加载数据
    print("="*50)
    print("TEXT-GCN训练")
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

    # 2. 构建图
    graph_builder = TextGraphBuilder(
        window_size=WINDOW_SIZE,
        min_freq=MIN_WORD_FREQ,
        pmi_threshold=PMI_THRESHOLD
    )

    # 构建训练图
    print("\n构建训练图...")
    train_data = graph_builder.build_text_graph(train_texts, train_labels, use_existing_vocab=False)
    train_data = train_data.to(device)

    # 构建测试图（使用相同的词汇表）
    print("\n构建测试图...")
    test_graph_builder = TextGraphBuilder(
        window_size=WINDOW_SIZE,
        min_freq=MIN_WORD_FREQ,
        pmi_threshold=PMI_THRESHOLD
    )

    # 复制训练图的词汇表和边
    test_graph_builder.vocab = graph_builder.vocab
    test_graph_builder.saved_edges = graph_builder.saved_edges
    test_graph_builder.saved_edge_weights = graph_builder.saved_edge_weights

    test_data = test_graph_builder.build_text_graph(test_texts, test_labels, use_existing_vocab=True)
    test_data = test_data.to(device)

    print(f"图构建完成:")
    print(f"- 总节点数: {train_data.x.shape[0]}")
    print(f"- 词节点数: {train_data.num_word_nodes}")
    print(f"- 文档节点数: {train_data.doc_mask.sum().item()}")
    print(f"- 边数: {train_data.edge_index.shape[1]}")

    # 3. 初始化模型
    model = TEXTGCN(
        input_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(model)

    # 4. 开始训练
    print("\n" + "="*50)
    print("开始训练...")

    train_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss = train_loop(model, train_data, optimizer, criterion)
        train_losses.append(train_loss)

        # 评估
        test_acc, test_prec, test_rec, test_f1 = eval_loop(model, test_data)
        test_accuracies.append(test_acc)
        test_precisions.append(test_prec)
        test_recalls.append(test_rec)
        test_f1s.append(test_f1)

        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        print(f"训练损失: {train_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print(f"测试F1分数: {test_f1:.4f}")
        print(f"测试精确率: {test_prec:.4f}")
        print(f"测试召回率: {test_rec:.4f}")
        print("-"*50)

    # 5. 计算平均结果
    avg_train_loss = np.mean(train_losses)
    avg_test_acc = np.mean(test_accuracies)
    avg_test_prec = np.mean(test_precisions)
    avg_test_rec = np.mean(test_recalls)
    avg_test_f1 = np.mean(test_f1s)

    best_f1_idx = np.argmax(test_f1s)
    best_acc_idx = np.argmax(test_accuracies)

    print("\n" + "="*50)
    print("TEXT-GCN训练完成！")
    print("="*50)
    print("训练结果统计:")
    print(f"平均训练损失: {avg_train_loss:.4f}")
    print(f"平均测试准确率: {avg_test_acc:.4f}")
    print(f"平均测试精确率: {avg_test_prec:.4f}")
    print(f"平均测试召回率: {avg_test_rec:.4f}")
    print(f"平均测试F1分数: {avg_test_f1:.4f}")
    print("\n最佳结果:")
    print(f"最高F1分数: {test_f1s[best_f1_idx]:.4f} (第{best_f1_idx+1}轮)")
    print(f"最高准确率: {test_accuracies[best_acc_idx]:.4f} (第{best_acc_idx+1}轮)")
    print("="*50)