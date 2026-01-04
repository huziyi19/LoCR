import numpy as np
import os
import json
import asyncio
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 导入项目模块
from text_to_graph import create_similarity_graph
from Logic_propagator import logic_propagation
from GCN_propagator import predict_missing_features
from split_and_detect import (
    async_split_and_detect_sentences,
    async_get_overall_confidence,
    async_batch_detect_sentences,
    clear_detection_cache
)

# --- 非线性放大函数 (来自原文件) ---
def nonlinear_amplify(delta, scale=10.0, bias=0):
    return 1 / (1 + np.exp(-scale * (delta - bias)))

def detect_and_amplify_changes_nonlinear(X, Y, threshold=0.5, scale=10.0):
    assert X.shape == Y.shape, "X and Y must have the same shape"
    delta = np.abs(Y - X)
    crossed = ((X < threshold) & (Y >= threshold)) | ((Y < threshold) & (X >= threshold))
    amplified = np.zeros_like(delta)
    amplified[crossed] = nonlinear_amplify(delta[crossed], scale=scale)
    amplified[~crossed] = delta[~crossed]
    return amplified

# === 实验方法函数 (严格隔离) ===

def run_llm_only_method(overall_confidence):
    """方法一：大模型综合判断"""
    return "fake" if overall_confidence < 0.5 else "real"

def run_gcn_method(adj, initial_confidence):
    """方法二：大模型 + GCN-Only"""
    gcn_feature = predict_missing_features(adj, initial_confidence.copy())
    return "fake" if np.any(gcn_feature < 0.5) else "real"

def run_logic_method(nodes, initial_confidence):
    """方法三：大模型 + Logic-Only"""
    n = len(nodes)
    if n <= 1:
        return "fake" if np.any(initial_confidence < 0.5) else "real"
        
    masked_confidence = initial_confidence.copy()
    mask_indices = np.random.choice(n, size=max(1, int(n * 0.3)), replace=False)
    masked_confidence[mask_indices] = np.nan
    
    logic_feature = logic_propagation(nodes, masked_confidence)
    
    nan_indices = np.isnan(logic_feature)
    logic_feature[nan_indices] = initial_confidence[nan_indices]
    
    return "fake" if np.any(logic_feature < 0.5) else "real"

async def run_full_lfnd_method(initial_confidence, nodes, adj, session_details):
    """方法四：大模型 + 完整 LFND"""
    n = len(nodes)
    if n == 0:
        return "real", 0 # 如果没有节点，无法判断，默认为real

    # LFND的正确逻辑：pre_feature 初始为空 (NaN)，只有修正的节点才拥有数值
    pre_feature_lfnd = np.full((n, 1), np.nan) 
    
    # initial_feature_lfnd 是基准，会随着修正而更新
    initial_feature_lfnd = initial_confidence.copy() 
    
    loop_numbers = 0
    final_feature_lfnd = initial_confidence.copy()

    # 第一次传播，使用完整的初始置信度
    l_feature = logic_propagation(nodes, initial_feature_lfnd)
    final_feature_lfnd = predict_missing_features(adj, l_feature)

    while True:
        gap = detect_and_amplify_changes_nonlinear(initial_feature_lfnd, final_feature_lfnd)
        wrong_nodes_index = np.argwhere(gap > 0.5)
        
        loop_numbers += 1
        if loop_numbers >= 3 or len(wrong_nodes_index) == 0:
            break

        wrong_sentences = [nodes[i[0]][1]['label'] for i in wrong_nodes_index]
        
        # 异步调用API进行修正
        correction = await async_batch_detect_sentences(
            wrong_sentences, **session_details
        )
        correction = correction.reshape(-1, 1)
        
        # 创建下一次迭代的 pre_feature，它只包含修正后的值
        pre_feature_lfnd = np.full((n, 1), np.nan)
        for idx, (node_idx, value) in enumerate(zip(wrong_nodes_index, correction)):
            pre_feature_lfnd[node_idx[0], 0] = value
            # 同时更新我们的基准，以便下一轮gap计算是基于最新的“事实”
            initial_feature_lfnd[node_idx[0], 0] = value
        
        # 用修正后的pre_feature进行下一次传播
        l_feature = logic_propagation(nodes, pre_feature_lfnd)
        final_feature_lfnd = predict_missing_features(adj, l_feature)

    pred = "fake" if np.any(final_feature_lfnd < 0.5) else "real"
    return pred, loop_numbers

# --- 实验配置 ---
BATCH_SIZE = 128
DATASETS_TO_TEST = {
    "gossipcop": os.path.join('data', 'gossipcop', 'test.jsonl'),
    "politifact": os.path.join('data', 'politifact', 'test.jsonl'),
}

# --- 主执行函数 ---
async def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: raise ValueError("请设置OPENROUTER_API_KEY环境变量")

    async_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    session_details = {
        "model_name": "deepseek/deepseek-chat", "api_key": api_key,
        "temperature": 0.3, "session": async_client
    }
    
    all_final_results = {}

    for dataset_name, dataset_path in DATASETS_TO_TEST.items():
        print(f"\n{'='*80}\n[处理数据集]: {dataset_name}\n{'='*80}")
        data_set = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'text' not in data and 'message' in data: data['text'] = data['message']
                    data_set.append(data)
        except FileNotFoundError:
            print(f"错误: 未找到数据集文件 {dataset_path}。请检查路径。")
            continue
        
        total_samples = len(data_set)
        if total_samples == 0: continue
            
        correct_counts = {'llm': 0, 'gcn': 0, 'logic': 0, 'lfnd': 0}
        total_lfnd_loops = 0
        start_time = time.time()

        for i in range(0, total_samples, BATCH_SIZE):
            batch_articles = data_set[i:i+BATCH_SIZE]
            print(f"\n--- 正在处理批次 {i//BATCH_SIZE + 1}/{(total_samples + BATCH_SIZE - 1)//BATCH_SIZE} (样本 {i+1}-{min(i+BATCH_SIZE, total_samples)}) ---")
            
            tasks_split = [async_split_and_detect_sentences(a['text'], **session_details) for a in batch_articles]
            results_split = await asyncio.gather(*tasks_split)
            
            tasks_overall = [async_get_overall_confidence(s, c, **session_details) for s, c in results_split]
            results_overall_confidence = await asyncio.gather(*tasks_overall)

            for j, article in enumerate(batch_articles):
                global_idx, sample_id, true_label = i + j + 1, article.get('id', 'N/A'), article['label']
                sentences, initial_confidence_list = results_split[j]

                # --- 运行检测方法 ---
                pred_llm = run_llm_only_method(results_overall_confidence[j])

                # 注释GCN和Logic方法，保留LFND方法
                if len(sentences) == 0:
                    # 默认值，用于没有句子的情况
                    pred_gcn = "real"
                    pred_logic = "real"
                    pred_lfnd, loop_numbers = "real", 0
                else:
                    initial_confidence = np.array(initial_confidence_list).reshape(-1, 1)
                    n = len(sentences)
                    G = create_similarity_graph(sentences)
                    nodes = list(G.nodes(data=True))
                    adj = np.zeros((n, n))
                    for u, v, w in G.edges(data=True): adj[u, v] = adj[v, u] = w['weight']

                    # 注释的GCN方法
                    # pred_gcn = run_gcn_method(adj, initial_confidence)
                    pred_gcn = "real"  # 默认值

                    # 注释的Logic方法
                    # pred_logic = run_logic_method(nodes, initial_confidence)
                    pred_logic = "real"  # 默认值

                    # 保留LFND方法
                    pred_lfnd, loop_numbers = await run_full_lfnd_method(initial_confidence, nodes, adj, session_details)
                
                # --- 统计与打印 ---
                if pred_llm == true_label: correct_counts['llm'] += 1
                if pred_gcn == true_label: correct_counts['gcn'] += 1
                if pred_logic == true_label: correct_counts['logic'] += 1
                if pred_lfnd == true_label: correct_counts['lfnd'] += 1
                total_lfnd_loops += loop_numbers
                clear_detection_cache()

                print(f"\n------------------------------------------------------------")
                print(f"样本 {global_idx}/{total_samples} (ID: {sample_id}) | 真实标签: {true_label}")
                res_llm = '✓ 正确' if pred_llm == true_label else '✗ 错误'
                res_gcn = '✓ 正确' if pred_gcn == true_label else '✗ 错误'
                res_logic = '✓ 正确' if pred_logic == true_label else '✗ 错误'
                res_lfnd = '✓ 正确' if pred_lfnd == true_label else '✗ 错误'
                print(f"  - 方法一 (LLM-Only):   预测 {pred_llm:<4} -> {res_llm}")
                print(f"  - 方法二 (LLM+GCN):    预测 {pred_gcn:<4} -> {res_gcn}")
                print(f"  - 方法三 (LLM+Logic):  预测 {pred_logic:<4} -> {res_logic}")
                print(f"  - 方法四 (Full LFND): 预测 {pred_lfnd:<4} -> {res_lfnd} (迭代 {loop_numbers} 轮)")
                print(f"------------------------------------------------------------")

        # --- 报告数据集结果 ---
        end_time = time.time()
        print(f"\n{'*'*30} 数据集 [{dataset_name}] 实验结果 {'*'*30}")
        print(f"总样本数: {total_samples}, 总耗时: {end_time - start_time:.2f} 秒")
        
        acc_llm = correct_counts['llm'] / total_samples
        acc_gcn = correct_counts['gcn'] / total_samples
        acc_logic = correct_counts['logic'] / total_samples
        acc_lfnd = correct_counts['lfnd'] / total_samples
        avg_consistency = total_lfnd_loops / total_samples

        print(f"  - 方法一 (大模型综合判断)  准确率: {acc_llm:.2%}")
        print(f"  - 方法二 (大模型 + GCN)     准确率: {acc_gcn:.2%}")
        print(f"  - 方法三 (大模型 + Logic)    准确率: {acc_logic:.2%}")
        print(f"  - 方法四 (大模型 + 完整LFND) 准确率: {acc_lfnd:.2%}")
        print(f"    - (LFND) 平均一致性 (迭代轮数): {avg_consistency:.2f}")
        
        all_final_results[dataset_name] = {
            'LLM-Only': f"{acc_llm:.2%}", 'LLM+GCN': f"{acc_gcn:.2%}",
            'LLM+Logic': f"{acc_logic:.2%}", 'Full_LFND': f"{acc_lfnd:.2%}",
            'LFND_Consistency': f"{avg_consistency:.2f}"
        }

    print(f"\n{'='*80}\n[所有实验完成]\n{'='*80}")
    print(json.dumps(all_final_results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())