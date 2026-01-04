import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def build_graph_from_adj(adj_matrix, features):
    edge_index = []
    edge_weight = []

    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0:
                edge_index.append([i, j])
                edge_weight.append(adj_matrix[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Replace NaNs with zeros for input
    feature_mask = ~np.isnan(features)
    features_filled = np.nan_to_num(features, nan=0.0)
    x = torch.tensor(features_filled, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), feature_mask


class GCNFeaturePropagator(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNFeaturePropagator, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return torch.sigmoid(x)  # 输出为 0~1 概率


def predict_missing_features(adj_matrix, features, hidden_dim=16, epochs=200, lr=0.01):
    data, feature_mask = build_graph_from_adj(adj_matrix, features)

    # 如果图中没有边，则无法进行GCN传播，直接返回填充后的特征
    if data.num_edges == 0:
        print("Warning: No edges in the graph, skipping GCN propagation.")
        # nan_to_num已经将nan替换为0，可以直接返回
        return data.x.numpy()
        
    model = GCNFeaturePropagator(in_channels=features.shape[1], hidden_channels=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # 添加权重衰减

    x = data.x
    known_mask = torch.tensor(feature_mask, dtype=torch.bool)

    # 检查是否有已知特征
    if not known_mask.any():
        print("Warning: No known features for training, using random initialization")
        # 如果没有已知特征，直接使用模型输出的初始值
        model.eval()
        with torch.no_grad():
            out = model(x, data.edge_index, data.edge_attr)
        return out.numpy()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, data.edge_index, data.edge_attr)

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss = F.mse_loss(out[known_mask], x[known_mask])

        # 检查loss是否为nan
        if torch.isnan(loss):
            print(f"Warning: NaN loss at epoch {epoch}, stopping training")
            break

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 用模型输出替代未知特征
    model.eval()
    with torch.no_grad():
        out = model(x, data.edge_index, data.edge_attr)
        final_features = x.clone()
        unknown_mask = ~known_mask
        final_features[unknown_mask] = out[unknown_mask]

    # 确保输出在0-1范围内
    final_features = torch.clamp(final_features, 0, 1)

    return final_features.numpy()

'''
# ==== 使用示例 ====

if __name__ == "__main__":
    # 示例图：4个节点，边权重表示相似度
    
    adj = np.array([
        [0.0, 0.8, 0.0, 0.0],
        [0.8, 0.0, 0.6, 0.0],
        [0.0, 0.6, 0.0, 0.9],
        [0.0, 0.0, 0.9, 0.0]
    ])

    # 节点特征：2维特征，部分为nan表示未知
    features = np.array([
        [1, 0],
        [np.nan, np.nan],
        [0, 1],
        [np.nan, np.nan]
    ])

    features = np.array([
        [0.89, 0.13],
        [np.nan, np.nan],
        [0.67, 0.93],
        [np.nan, np.nan]
    ])
    
    adj = np.array([
        [0.0, 0.8, 0.0, 0.0],
        [0.8, 0.0, 0.6, 0.0],
        [0.0, 0.6, 0.0, 0.9],
        [0.0, 0.0, 0.9, 0.0]
    ])
    features = np.array([
        [0.89],
        [np.nan],
        [0.67],
        [np.nan]
    ])

    predicted = predict_missing_features(adj, features)
    print("Predicted features:")
    print(predicted)
'''

