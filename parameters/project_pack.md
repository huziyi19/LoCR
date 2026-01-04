# Project Files Pack

Generated on: 2025-10-07 18:00:39

Base directory: /data01/yangkun/yjy/new_LFND/new_try/deepseekv3


---

## File: `.env`

Size: 555.0 B


```

# # --- Bocha AI Configuration (for Search) ---
# BOCHA_AI_KEY="sk-12ee47867235426d949bb9ec0413bd5d"
# BOCHA_SEARCH_API_URL="https://api.bochaai.com/v1/web-search"

# --- OpenRouter & LLM Configuration ---
OPENROUTER_API_KEY="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"
OPENROUTER_MODEL="deepseek/deepseek-chat"
OPENROUTER_API_URL="https://openrouter.ai/api/v1/chat/completions"

# # [可选] OpenRouter推荐的请求头信息
# OPENROUTER_REFERER="https://github.com"
# OPENROUTER_APP_NAME="Fact-Checking Bot"


```

---

## File: `GCN_propagator.py`

Size: 3.9 KB


```python

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
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, batch.edge_index, batch.edge_attr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss = F.mse_loss(out[known_mask], x[known_mask])
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch}, stopping training")
                break
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        final_out = model(x, batch.edge_index, batch.edge_attr)
        final_features = x.clone()
        unknown_mask = ~known_mask
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
            # print("Warning: No edges in the graph, skipping GCN propagation.") # <--- 再次注释掉这一行
            output.append(d.x.numpy())
            
    return output

```

---

## File: `LFND.py`

Size: 10.2 KB


```python

import numpy as np
import os
import json
import asyncio
import time
from itertools import product
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 导入项目模块
from text_to_graph import create_similarity_graph
from Logic_propagator import logic_propagation
from GCN_propagator import build_graph_from_adj, predict_missing_features_batch
from split_and_detect import (
    async_split_and_detect_sentences,
    async_get_overall_confidence,
    async_batch_detect_sentences,
    clear_detection_cache
)

# --- 非线性放大函数 ---
def nonlinear_amplify(delta, scale=10.0, bias=0):
    return 1 / (1 + np.exp(-scale * (delta - bias)))

def detect_and_amplify_changes_nonlinear(X, Y, threshold=0.5, scale=10.0):
    if X.shape != Y.shape: return np.array([0])
    delta = np.abs(Y - X)
    crossed = ((X < threshold) & (Y >= threshold)) | ((Y < threshold) & (X >= threshold))
    amplified = np.zeros_like(delta)
    amplified[crossed] = nonlinear_amplify(delta[crossed], scale=scale)
    amplified[~crossed] = delta[~crossed]
    return amplified

# === 并行执行器函数 ===
async def run_all_lfnd_configs_parallel(initial_confidence, sentences, session_details, configs):
    n = len(sentences)
    if n == 0:
        return [("real", 0) for _ in configs]

    states = []
    for config in configs:
        sim_thresh, logic_thresh = config["sim_thresh"], config["logic_thresh"]
        G = create_similarity_graph(sentences, threshold=sim_thresh)
        nodes = list(G.nodes(data=True))
        adj = np.zeros((n, n))
        for u, v, w in G.edges(data=True): adj[u, v] = adj[v, u] = w['weight']
        
        states.append({
            "config": config, "nodes": nodes, "adj": adj,
            "pre_feature": initial_confidence.copy(),
            "initial_feature": initial_confidence.copy(),
            "final_feature": initial_confidence.copy(),
            "loop_number": 0, "is_done": False,
        })

    for i in range(3):
        if all(s['is_done'] for s in states): break

        active_states = [s for s in states if not s['is_done']]
        
        l_features_for_gcn = [logic_propagation(
            s['nodes'], s['pre_feature'], s['config']['logic_thresh']
        ) for s in active_states]

        gcn_inputs = [build_graph_from_adj(s['adj'], l) for s, l in zip(active_states, l_features_for_gcn)]
        gcn_results = predict_missing_features_batch(gcn_inputs)
        
        for idx, state in enumerate(active_states):
             if idx < len(gcn_results):
                state['final_feature'] = gcn_results[idx]

        all_wrong_sentences_map = {}
        states_needing_correction = []
        
        for state in active_states:
            state['loop_number'] += 1
            gap = detect_and_amplify_changes_nonlinear(state['initial_feature'], state['final_feature'])
            wrong_nodes_indices = np.argwhere(gap > 0.5)

            if state['loop_number'] >= 3 or len(wrong_nodes_indices) == 0:
                state['is_done'] = True
                continue

            states_needing_correction.append(state)
            state['wrong_nodes_indices'] = wrong_nodes_indices
            for node_idx in wrong_nodes_indices:
                sentence = state['nodes'][node_idx[0]][1]['label']
                all_wrong_sentences_map[(sentence, id(state))] = None
        
        if all_wrong_sentences_map:
            unique_sentences = list(dict.fromkeys([s for s, i in all_wrong_sentences_map.keys()]))
            corrections = await async_batch_detect_sentences(unique_sentences, **session_details)
            correction_map = {sent: corr for sent, corr in zip(unique_sentences, corrections)}

            for state in states_needing_correction:
                state['pre_feature'] = np.full((n, 1), np.nan)
                for node_idx in state['wrong_nodes_indices']:
                    sent = state['nodes'][node_idx[0]][1]['label']
                    value = correction_map.get(sent, 0.5)
                    state['pre_feature'][node_idx[0], 0] = value
                    state['initial_feature'][node_idx[0], 0] = value
        
        for state in active_states:
            if not state['is_done'] and state not in states_needing_correction:
                 state['is_done'] = True

    final_results = []
    for state in states:
        pred = "fake" if np.any(state['final_feature'] < 0.5) else "real"
        final_results.append((pred, state['loop_number']))
    return final_results

# --- 实验配置 ---
BATCH_SIZE = 128
DATASETS_TO_TEST = {
    "gossipcop": os.path.join('data', 'gossipcop', 'test.jsonl'),
    "politifact": os.path.join('data', 'politifact', 'test.jsonl'),
}
BASELINE_PARAMS = {"sim_thresh": 0.5, "logic_thresh": 0.8}
SINGLE_VAR_PARAMS = [
    {"sim_thresh": 0.6, "logic_thresh": 0.8},
    {"sim_thresh": 0.8, "logic_thresh": 0.8},
    {"sim_thresh": 0.5, "logic_thresh": 0.7},
    {"sim_thresh": 0.5, "logic_thresh": 0.9},
]
ALL_LFND_CONFIGS = [BASELINE_PARAMS] + SINGLE_VAR_PARAMS

# --- 主执行函数 ---
async def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    MODEL_NAME = os.getenv("OPENROUTER_MODEL")
    if not api_key: raise ValueError("请在 .env 文件中设置 OPENROUTER_API_KEY")
    if not MODEL_NAME: raise ValueError("请在 .env 文件中设置 OPENROUTER_MODEL")

    async_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    session_details = { "model_name": MODEL_NAME, "api_key": api_key, "temperature": 0.3, "session": async_client }
    print(f"[模型配置] 当前使用模型: {MODEL_NAME}")
    
    all_final_results = {}
    for dataset_name, dataset_path in DATASETS_TO_TEST.items():
        print(f"\n{'='*80}\n[处理数据集]: {dataset_name}\n{'='*80}")
        data_set = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f: data_set.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"错误: 未找到数据集文件 {dataset_path}。"); continue
        
        total_samples = len(data_set)
        if total_samples == 0: continue
            
        llm_correct_count = 0
        lfnd_correct_counts = {tuple(c.values()): 0 for c in ALL_LFND_CONFIGS}
        lfnd_total_loops = {tuple(c.values()): 0 for c in ALL_LFND_CONFIGS}
        
        start_time = time.time()
        for i in range(0, total_samples, BATCH_SIZE):
            batch_articles = data_set[i:i+BATCH_SIZE]
            print(f"\n--- 正在处理批次 {i//BATCH_SIZE + 1}/{(total_samples + BATCH_SIZE - 1)//BATCH_SIZE} (样本 {i+1}-{min(i+BATCH_SIZE, total_samples)}) ---")
            
            tasks_split = [async_split_and_detect_sentences(a.get('text', a.get('message')), **session_details) for a in batch_articles]
            results_split = await asyncio.gather(*tasks_split)
            
            tasks_overall = [async_get_overall_confidence(s, c, **session_details) for s, c in results_split]
            results_overall_confidence = await asyncio.gather(*tasks_overall)

            for j, article in enumerate(batch_articles):
                global_idx, sample_id, true_label = i + j + 1, article.get('id', 'N/A'), article['label']
                sentences, initial_confidence_list = results_split[j]

                pred_llm = "fake" if results_overall_confidence[j] < 0.5 else "real"
                if pred_llm == true_label: llm_correct_count += 1
                
                print(f"\n------------------------------------------------------------")
                print(f"样本 {global_idx}/{total_samples} (ID: {sample_id}) | 真实标签: {true_label}")
                res_llm = '✓ 正确' if pred_llm == true_label else '✗ 错误'
                print(f"  - 方法 (LLM-Only):   预测 {pred_llm:<4} -> {res_llm}")

                if len(sentences) > 0:
                    initial_confidence = np.array(initial_confidence_list).reshape(-1, 1)
                    print(f"  - LFND 超参数分析:")
                    
                    lfnd_results = await run_all_lfnd_configs_parallel(initial_confidence, sentences, session_details, ALL_LFND_CONFIGS)
                    
                    for idx, config in enumerate(ALL_LFND_CONFIGS):
                        pred_lfnd, loop_numbers = lfnd_results[idx]
                        param_key = tuple(config.values())
                        if pred_lfnd == true_label: lfnd_correct_counts[param_key] += 1
                        lfnd_total_loops[param_key] += loop_numbers
                        
                        res_lfnd = '✓ 正确' if pred_lfnd == true_label else '✗ 错误'
                        print(f"    - (Sim_Thresh={config['sim_thresh']}, Logic_Thresh={config['logic_thresh']}): 预测 {pred_lfnd:<4} -> {res_lfnd} (迭代 {loop_numbers} 轮)")

                print(f"------------------------------------------------------------")
                clear_detection_cache()

        end_time = time.time()
        print(f"\n{'*'*30} 数据集 [{dataset_name}] 实验结果 {'*'*30}")
        print(f"总样本数: {total_samples}, 总耗时: {end_time - start_time:.2f} 秒")
        
        acc_llm = llm_correct_count / total_samples
        print(f"\n- 基准模型:"); print(f"  - LLM-Only 准确率: {acc_llm:.2%}")
        print(f"\n- LFND 模型超参数分析:")
        dataset_results = {'LLM-Only': f"{acc_llm:.2%}"}
        
        config_groups = {
            "初始配置": [BASELINE_PARAMS],
            "单一变量对照": SINGLE_VAR_PARAMS,
        }
        for group_name, configs in config_groups.items():
            print(f"\n  - {group_name}:")
            for config in configs:
                s, l = config["sim_thresh"], config["logic_thresh"]
                acc = lfnd_correct_counts[(s, l)] / total_samples
                cons = lfnd_total_loops[(s, l)] / total_samples
                print(f"    - (Sim_Thresh={s}, Logic_Thresh={l}): 准确率: {acc:.2%}, 平均一致性: {cons:.2f} 轮")
                dataset_results[f"LFND_Sim{s}_Logic{l}"] = {"accuracy": f"{acc:.2%}", "consistency": f"{cons:.2f}"}
        all_final_results[dataset_name] = dataset_results

    print(f"\n{'='*80}\n[所有实验完成]\n{'='*80}")
    print(json.dumps(all_final_results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

```

---

## File: `Logic_propagator.py`

Size: 3.0 KB


```python

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import pipeline
from itertools import combinations
import torch
import re
import difflib
import numpy as np

NEGATION_WORDS = {"not", "no", "never", "n't", "didn't", "doesn't", "don't", "cannot", "won't", "wasn't", "isn't"}

# 加载自然语言推理模型
nli_pipeline = pipeline("text-classification", model="roberta-large-mnli")

def tokenize(sentence):
    """小写、去标点、分词"""
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.split()

def is_logical_negation(sent1, sent2, min_similarity=0.6):
    """判断 sent1 和 sent2 是否互为“非”"""
    words1, words2 = tokenize(sent1), tokenize(sent2)
    neg1 = any(w in NEGATION_WORDS for w in words1)
    neg2 = any(w in NEGATION_WORDS for w in words2)
    if neg1 == neg2: return False
    clean1 = [w for w in words1 if w not in NEGATION_WORDS]
    clean2 = [w for w in words2 if w not in NEGATION_WORDS]
    str1, str2 = ' '.join(clean1), ' '.join(clean2)
    similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
    return similarity >= min_similarity

def extract_logical_edges(nodes, score_threshold=0.8):
    edges = []
    for (i1, label1), (i2, label2) in combinations(nodes, 2):
        result1 = nli_pipeline((label1['label'], label2['label']), padding=True, truncation=True)
        result2 = nli_pipeline((label2['label'], label1['label']), padding=True, truncation=True)

        if result1['label'] == 'ENTAILMENT' and result1['score'] > score_threshold:
            edges.append((i1, i2, {'label': 'imply'}))
        elif result1['label'] == 'CONTRADICTION' and result1['score'] > score_threshold:
            if is_logical_negation(label1['label'], label2['label']):
                edges.append((i1, i2, {'label': 'not'}))

        if result2['label'] == 'ENTAILMENT' and result2['score'] > score_threshold:
            edges.append((i2, i1, {'label': 'imply'}))
        elif result2['label'] == 'CONTRADICTION' and result2['score'] > score_threshold:
            if is_logical_negation(label1['label'], label2['label']):
                edges.append((i2, i1, {'label': 'not'}))
    return edges

def logic_propagation(nodes, feature_matrix, logic_score_threshold=0.8):
    n = len(nodes)
    f_matrix = feature_matrix.copy()
    edges = extract_logical_edges(nodes, score_threshold=logic_score_threshold)

    for i, j, logic in edges:
        if logic['label'] == 'not':
            if np.isnan(feature_matrix[i,0]) and (not np.isnan(feature_matrix[j,0])):
                f_matrix[i,0] = 1 - feature_matrix[j,0]
            elif np.isnan(feature_matrix[j,0]) and (not np.isnan(feature_matrix[i,0])):
                f_matrix[j,0] = 1 - feature_matrix[i,0]
        if logic['label'] == 'imply':
            if (not np.isnan(feature_matrix[i,0])) and np.isnan(feature_matrix[j,0]):
                f_matrix[j,0] = feature_matrix[i,0]
    return f_matrix

```

---

## File: `load_data.py`

Size: 2.3 KB


```python

import os
import jsonlines
import csv

def normalization(dataset_name, file_path):
    """
    从指定的单个文件路径加载和归一化数据。
    能自动识别 .jsonl 和 .csv 文件。
    """
    print(f"正在从文件 '{file_path}' 加载数据集 '{dataset_name}'...")
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件路径 {file_path}。")
        return []

    # 根据文件扩展名选择合适的读取函数
    file_extension = os.path.splitext(file_path)[1].lower()
    
    dataset = []
    try:
        if file_extension == '.jsonl':
            with jsonlines.open(file_path, mode='r') as reader:
                for idx, item in enumerate(reader):
                    dataset.append({
                        'id': item.get('id', str(idx)),
                        'text': item.get('text', '') or item.get('message', ''), # 兼容不同字段名
                        'label': item.get('label', ''),
                        'title': item.get('title', '')
                    })
        elif file_extension == '.csv':
            with open(file_path, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for idx, row in enumerate(reader):
                    # 兼容 welfake 数据集的标签 (1/0)
                    label = row.get('label', '')
                    if label == '1': label = 'real'
                    if label == '0': label = 'fake'
                    
                    dataset.append({
                        'id': row.get('id', str(idx)),
                        'text': row.get('text', '') or row.get('content', ''), # 兼容不同字段名
                        'label': label,
                        'title': row.get('title', '')
                    })
        else:
            print(f"错误：不支持的文件类型 '{file_extension}'。请提供 .jsonl 或 .csv 文件。")
            return []

    except Exception as e:
        print(f"读取或解析 {file_path} 时出错: {e}")
        return []

    # 过滤掉内容或标签为空的无效数据
    valid_dataset = [item for item in dataset if item.get('text') and item.get('label')]
    print(f"成功加载并过滤后，得到 {len(valid_dataset)} 条有效数据。")
    return valid_dataset

```

---

## File: `requirements.txt`

Size: 172.0 B


```text

ÿþt o r c h 
 
 n l t k 
 
 s p a c y 
 
 s e n t e n c e - t r a n s f o r m e r s   n u m p y 
 
 p y y a m l 
 
 j s o n l i n e s 
 
 n e t w o r k x 
 
 d i f f l i b 

```

---

## File: `split_and_detect.py`

Size: 6.9 KB


```python

import os
import numpy as np
import asyncio
import re
from openai import OpenAI, AsyncOpenAI
from openai import APIError
import json
from dotenv import load_dotenv

load_dotenv()
detection_cache = {}

def clear_detection_cache():
    global detection_cache
    detection_cache.clear()

async def async_split_and_detect_sentences(text, model_name, api_key, temperature, session: AsyncOpenAI):
    prompt = f"""Please split the following news text into multiple, semantically complete sentences. For each sentence, you must assess its truthfulness confidence score.

Requirements:
1. When splitting, maintain the full semantic meaning of each sentence. Avoid over-splitting.
2. Generally, split at natural sentence breaks like periods, question marks, or exclamation marks.
3. For each sentence, provide a truthfulness confidence score (a decimal value between 0.00 and 1.00).
4. The confidence score represents the likelihood that the statement is true.
5. Output format: One sentence per line, in the format "Sentence content [0.xx]".
6. Adhere strictly to this format. Do not add numbers, bullet points, or any other markers.

News Text:
{text}

Begin splitting and assessing:"""
    try:
        response = await session.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000
        )
        result = response.choices[0].message.content
        sentences, confidences = [], []
        lines = result.strip().split('\n')
        for line in lines:
            line = line.strip()
            if '[' in line and ']' in line:
                last_bracket, first_bracket = line.rfind(']'), line.rfind('[')
                if first_bracket != -1 and last_bracket != -1:
                    sentence = line[:first_bracket].strip()
                    confidence_str = line[first_bracket+1:last_bracket].strip()
                    try:
                        confidence = float(re.split(r'\s+', confidence_str)[0])
                        sentences.append(sentence)
                        confidences.append(confidence)
                    except (ValueError, IndexError):
                        sentences.append(sentence)
                        confidences.append(0.5)
        if not sentences:
             sentences = [s.strip() for s in text.split('.') if s.strip()]
             confidences = [0.5] * len(sentences)
        return sentences, confidences
    except (APIError, json.JSONDecodeError) as e:
        print(f"!!! 异步API调用解析错误: {type(e).__name__} - {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"--- 服务器原始响应 ---\n{e.response.text[:500]}...\n--------------------")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        confidences = [0.5] * len(sentences)
        return sentences, confidences
    except Exception as e:
        print(f"!!! 未知异步API错误: {type(e).__name__} - {e}")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        confidences = [0.5] * len(sentences)
        return sentences, confidences

async def async_get_overall_confidence(sentences, confidences, model_name, api_key, temperature, session: AsyncOpenAI):
    if not sentences: return 0.5
    analysis_str = "\n".join([f"{i+1}. {s} [Confidence: {c:.2f}]" for i, (s, c) in enumerate(zip(sentences, confidences))])
    prompt = f"""You are a meticulous fact-checking analyst.

You have already broken down a news article into the following core sentences and assessed an initial truthfulness confidence score for each.

Analysis Results:
{analysis_str}

Now, considering all the sentences and their respective confidence scores, provide a final, overall truthfulness confidence score for the **entire news article** (a single decimal value between 0.0 and 1.0).

Please output only the final confidence number, without any extra explanations or text."""
    try:
        response = await session.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        match = re.search(r"(\d\.?\d*)", result)
        return float(match.group(1)) if match else 0.5
    except (APIError, json.JSONDecodeError) as e:
        print(f"!!! 获取总体置信度API解析错误: {type(e).__name__} - {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"--- 服务器原始响应 ---\n{e.response.text[:500]}...\n--------------------")
        return 0.5
    except Exception as e:
        print(f"!!! 未知获取总体置信度API错误: {type(e).__name__} - {e}")
        return 0.5

async def async_batch_detect_sentences(sentences, model_name, api_key, temperature, session: AsyncOpenAI):
    if not sentences: return np.array([])
    prompt = f"""Please assess the truthfulness confidence score (a decimal value between 0.0 and 1.0) for each of the following sentences.

Requirements:
1. Output format: One sentence per line, in the format "Sentence content [confidence score]".
2. The confidence score represents the likelihood that the statement is true.
3. Remain objective and neutral, basing your judgment on common knowledge.

Sentences to assess:
"""
    for i, sent in enumerate(sentences, 1):
        prompt += f"{i}. {sent}\n"
    prompt += "\nPlease give your assessment:"
    try:
        response = await session.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000
        )
        result = response.choices[0].message.content
        confidences = []
        lines = result.strip().split('\n')
        for line in lines:
            if '[' in line and ']' in line:
                parts = line.rsplit('[', 1)
                if len(parts) == 2:
                    try:
                        conf = float(parts[1].rstrip(']').strip())
                        confidences.append(conf)
                    except:
                        confidences.append(0.5)
        while len(confidences) < len(sentences):
            confidences.append(0.5)
        return np.array(confidences[:len(sentences)])
    except (APIError, json.JSONDecodeError) as e:
        print(f"!!! 异步批量检测API解析错误: {type(e).__name__} - {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"--- 服务器原始响应 ---\n{e.response.text[:500]}...\n--------------------")
        return np.array([0.5] * len(sentences))
    except Exception as e:
        print(f"!!! 未知异步批量检测API错误: {type(e).__name__} - {e}")
        return np.array([0.5] * len(sentences))

```

---

## File: `text_to_graph.py`

Size: 970.0 B


```python

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

```