import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
from sentence_transformers import SentenceTransformer, util
import networkx as nx

# 加载预训练的句子嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_similarity_graph(statement_set, threshold=0.5):
    # 将语句集合转换为嵌入向量
    embeddings = model.encode(statement_set, convert_to_numpy=True)

    # 计算所有语句对之间的余弦相似度
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    for i, statement in enumerate(statement_set):
        G.add_node(i, label=statement)

    # 添加边
    for i in range(len(statement_set)):
        for j in range(i + 1, len(statement_set)):
            similarity = similarity_matrix[i][j].item()
            if similarity > threshold:
                G.add_edge(i, j, weight=similarity)

    return G
