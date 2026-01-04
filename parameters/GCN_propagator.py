import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
        return torch.sigmoid(x)

def predict_missing_features(adj_matrix, features, hidden_dim=16, epochs=200, lr=0.01):
    data, feature_mask = build_graph_from_adj(adj_matrix, features)
    if data.num_edges == 0:
        print("Warning: No edges in the graph, skipping GCN propagation.")
        return data.x.numpy()
    
    results = predict_missing_features_batch([(data, feature_mask)], hidden_dim, epochs, lr)
    return results[0]

def predict_missing_features_batch(data_and_masks, hidden_dim=16, epochs=200, lr=0.01):
    valid_data_and_masks = [(d, m) for d, m in data_and_masks if d.num_edges > 0]
    if not valid_data_and_masks:
        return [d.x.numpy() for d, m in data_and_masks]

    data_list, mask_list = zip(*valid_data_and_masks)

    loader = DataLoader(list(data_list), batch_size=len(data_list))
    batch = next(iter(loader))

    in_channels = batch.num_features

    # 大改：动态调整隐藏层维度和学习率
    edge_count = batch.num_edges
    node_count = batch.num_nodes
    edge_density = edge_count / (node_count * (node_count - 1) / 2) if node_count > 1 else 0

    # 根据图的密度调整模型复杂度
    if edge_density > 0.3:
        hidden_dim = max(32, int(hidden_dim * 2))
        lr = 0.02
    elif edge_density > 0.1:
        hidden_dim = max(24, int(hidden_dim * 1.5))
        lr = 0.015
    else:
        hidden_dim = 16
        lr = 0.01

    model = GCNFeaturePropagator(in_channels, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    x = batch.x
    known_mask = torch.tensor(np.concatenate([m for m in mask_list]), dtype=torch.bool)

    if not known_mask.any():
        print("Warning: No known features in batch for training.")
        model.eval()
        with torch.no_grad():
            out = model(x, batch.edge_index, batch.edge_attr)
    else:
        # 大改：调整训练策略
        effective_epochs = min(epochs, max(50, int(epochs * (1 + edge_density))))

        for epoch in range(effective_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, batch.edge_index, batch.edge_attr)

            # 动态调整损失函数权重
            base_loss = F.mse_loss(out[known_mask], x[known_mask])

            # 添加边权重相关的正则化
            if batch.edge_attr is not None and len(batch.edge_attr) > 0:
                edge_weight_loss = torch.mean(batch.edge_attr) * 0.1
                total_loss = base_loss + edge_weight_loss
            else:
                total_loss = base_loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if torch.isnan(total_loss):
                print(f"Warning: NaN loss at epoch {epoch}, stopping training")
                break
            total_loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        final_out = model(x, batch.edge_index, batch.edge_attr)
        final_features = x.clone()
        unknown_mask = ~known_mask

        # 大改：根据图密度调整传播强度
        if edge_density > 0.2:
            # 高密度图：更激进的传播
            propagation_strength = min(1.5, 1 + edge_density)
            final_out = final_out * propagation_strength
        elif edge_density > 0.05:
            # 中等密度图：标准传播
            propagation_strength = 1.0
        else:
            # 低密度图：保守传播
            propagation_strength = max(0.7, 0.8 + edge_density * 4)
            final_out = final_out * propagation_strength

        final_features[unknown_mask] = final_out[unknown_mask]

    final_features = torch.clamp(final_features, 0, 1)

    split_sizes = [d.num_nodes for d in data_list]
    split_results = torch.split(final_features, split_sizes)

    final_results_map = {id(d): r.numpy() for d, r in zip(data_list, split_results)}

    output = []
    for d, m in data_and_masks:
        if id(d) in final_results_map:
            output.append(final_results_map[id(d)])
        else:
            output.append(d.x.numpy())

    return output