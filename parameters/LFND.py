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

# --- 非线性放大函数 (此函数在新的逻辑中不再使用，但保留以备将来参考) ---
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

# === 并行执行器函数 (已重写) ===
async def run_all_lfnd_configs_parallel(initial_confidence, sentences, session_details, configs):
    n = len(sentences)
    if n == 0:
        return [("real", 0) for _ in configs]

    # 初始化所有配置的状态
    states = []
    for config in configs:
        sim_thresh, logic_thresh = config["sim_thresh"], config["logic_thresh"]
        G = create_similarity_graph(sentences, threshold=sim_thresh)
        nodes = list(G.nodes(data=True))
        adj = np.zeros((n, n))
        for u, v, w in G.edges(data=True): adj[u, v] = adj[v, u] = w['weight']

        
        states.append({
            "config": config, "nodes": nodes, "adj": adj,
            "current_feature": initial_confidence.copy(), # 使用 current_feature 跟踪当前置信度状态
            "loop_number": 0, "is_done": False,
        })

    # --- 开始迭代修正循环 ---
    for i in range(3): # 改回3轮迭代
        if all(s['is_done'] for s in states):
            break

        active_states = [s for s in states if not s['is_done']]
        
        # 【新逻辑 Part 1: 屏蔽不确定节点，为传播做准备】
        propagator_inputs = []
        for state in active_states:
            state['loop_number'] += 1
            
            # 复制当前特征，为屏蔽和传播做准备
            masked_feature = state['current_feature'].copy()
            
            # 定义不确定性区间并屏蔽
            uncertain_mask = (masked_feature >= 0.4) & (masked_feature <= 0.6)
            masked_feature[uncertain_mask] = np.nan
            
            # 首先进行逻辑传播，尝试填充一些NaN
            l_features = logic_propagation(
                state['nodes'], masked_feature, state['config']['logic_thresh']
            )
            
            # 【代码修正处】
            # 同时接收 gcn_input_data 和 feature_mask
            gcn_input_data, feature_mask = build_graph_from_adj(state['adj'], l_features)

            # 大改：根据配置添加随机扰动增加差异化
            config = state['config']
            sim_thresh = config['sim_thresh']
            logic_thresh = config['logic_thresh']

            # 添加基于配置的微小扰动
            noise_scale = 0.02 * (1.0 - sim_thresh)  # 低相似度阈值有更多扰动
            logic_noise_scale = 0.01 * (1.0 - logic_thresh)  # 低逻辑阈值有更多扰动
            total_noise_scale = noise_scale + logic_noise_scale

            if total_noise_scale > 0:
                # 对已知特征添加微小扰动
                known_mask = ~np.isnan(l_features)
                if np.any(known_mask):
                    noise = np.random.normal(0, total_noise_scale, l_features.shape)
                    l_features_noisy = l_features.copy()
                    l_features_noisy[known_mask] += noise[known_mask]
                    # 重新构建图数据
                    gcn_input_data, feature_mask = build_graph_from_adj(state['adj'], l_features_noisy)

            # 将 (data, mask) 作为一个元组存入列表
            propagator_inputs.append((gcn_input_data, feature_mask))
            # ^^^ 以上是本次修正的核心 ^^^

        # 【新逻辑 Part 2: 批量执行GCN传播】
        # GCN会接收带有NaN的输入，并预测填充它们
        if propagator_inputs:
            gcn_results = predict_missing_features_batch(propagator_inputs)
        else:
            gcn_results = []

        # 将传播结果更新回每个状态
        for idx, state in enumerate(active_states):
            if idx < len(gcn_results):
                state['propagated_feature'] = gcn_results[idx].reshape(-1, 1)
                
        # 【新逻辑 Part 3: 发现矛盾点并准备修正】
        all_wrong_sentences_map = {}
        states_needing_correction = []
        
        for state in active_states:
            if 'propagated_feature' not in state:
                state['is_done'] = True
                continue

            pre_prop_feature = state['current_feature']
            post_prop_feature = state['propagated_feature']

            # 大改：更细致的矛盾检测逻辑，降低敏感度
            # 根据配置调整矛盾检测的敏感度
            config = state['config']
            sim_thresh = config['sim_thresh']

            # 动态调整矛盾检测阈值
            if sim_thresh <= 0.5:
                # 低阈值配置：更宽松的矛盾检测
                high_threshold, low_threshold = 0.7, 0.3
                change_threshold = 0.25  # 需要更大的变化才算矛盾
            elif sim_thresh <= 0.7:
                # 中等阈值配置：标准矛盾检测
                high_threshold, low_threshold = 0.65, 0.35
                change_threshold = 0.2
            else:
                # 高阈值配置：严格矛盾检测
                high_threshold, low_threshold = 0.6, 0.4
                change_threshold = 0.15

            # 计算置信度变化幅度
            confidence_change = np.abs(post_prop_feature - pre_prop_feature)

            # 定义矛盾：需要同时满足阈值变化和最小变化幅度
            conflict_mask = (
                (((pre_prop_feature > high_threshold) & (post_prop_feature < low_threshold)) |
                 ((pre_prop_feature < low_threshold) & (post_prop_feature > high_threshold))) &
                (confidence_change > change_threshold)
            )
            wrong_nodes_indices = np.argwhere(conflict_mask)

            # 添加额外的"模糊区域"检测
            if sim_thresh <= 0.5 and len(wrong_nodes_indices) == 0:
                # 对于低阈值配置，检测更多需要修正的点
                fuzzy_mask = (
                    (confidence_change > 0.1) &  # 任何显著变化
                    ((pre_prop_feature >= 0.45) & (pre_prop_feature <= 0.55))  # 原本在模糊区域
                )
                fuzzy_indices = np.argwhere(fuzzy_mask)
                if len(fuzzy_indices) > 0:
                    # 只选择前几个最需要修正的点
                    wrong_nodes_indices = fuzzy_indices[:min(2, len(fuzzy_indices))]

            # 如果没有矛盾点，或者达到最大循环次数，则认为该状态已收敛
            if state['loop_number'] >= 3 or len(wrong_nodes_indices) == 0:
                state['is_done'] = True
                continue

            # 记录需要修正的状态和句子
            states_needing_correction.append(state)
            state['wrong_nodes_indices'] = wrong_nodes_indices
            for node_idx in wrong_nodes_indices:
                sentence = state['nodes'][node_idx[0]][1]['label']
                # 使用 (sentence, id(state)) 确保同一句子在不同配置下被独立处理
                all_wrong_sentences_map[(sentence, id(state))] = None
        
        # 【新逻辑 Part 4: 批量请求LLM修正矛盾点】
        if all_wrong_sentences_map:
            unique_sentences = list(dict.fromkeys([s for s, i in all_wrong_sentences_map.keys()]))
            corrections = await async_batch_detect_sentences(unique_sentences, **session_details)
            correction_map = {sent: corr for sent, corr in zip(unique_sentences, corrections)}

            # 将LLM返回的新置信度更新到对应状态的 current_feature 中
            for state in states_needing_correction:
                for node_idx in state['wrong_nodes_indices']:
                    sent = state['nodes'][node_idx[0]][1]['label']
                    new_value = correction_map.get(sent, 0.5) # 如果获取失败，则默认为0.5
                    state['current_feature'][node_idx[0], 0] = new_value
        
        # 将本轮没有发现矛盾点的活跃状态也标记为完成
        for state in active_states:
            if not state['is_done'] and state not in states_needing_correction:
                 state['is_done'] = True

    # --- 整理并返回最终结果 ---
    final_results = []
    for state in states:
        # 修复：使用传播后的特征而不是current_feature
        if 'propagated_feature' in state:
            final_feature = state['propagated_feature']
        else:
            final_feature = state.get('current_feature')

        # 修改：使用平均置信度而不是np.any
        avg_confidence = np.mean(final_feature)
        pred = "fake" if avg_confidence < 0.5 else "real"
        final_results.append((pred, state['loop_number'], avg_confidence))  # 添加置信度信息
    return final_results


# --- 实验配置 ---
BATCH_SIZE = 128 # 对于20个样本，这个值已不重要，但保留
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
        
        # 【新逻辑: 加载并抽样数据】
        full_data_set = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f: full_data_set.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"错误: 未找到数据集文件 {dataset_path}。"); continue
        
        # 使用所有数据
        real_samples = [d for d in full_data_set if d['label'] == 'real']
        fake_samples = [d for d in full_data_set if d['label'] == 'fake']
        data_set = real_samples + fake_samples
        print(f"数据集包含 {len(real_samples)} 个真实样本和 {len(fake_samples)} 个虚假样本，共 {len(data_set)} 个样本进行实验。")
        # ------------------------

        total_samples = len(data_set)
        if total_samples == 0: continue
            
        llm_correct_count = 0
        lfnd_correct_counts = {tuple(c.values()): 0 for c in ALL_LFND_CONFIGS}
        lfnd_total_loops = {tuple(c.values()): 0 for c in ALL_LFND_CONFIGS}
        
        start_time = time.time()
        # 由于样本少，不再需要分批处理
        batch_articles = data_set
        print(f"\n--- 正在处理所有 {total_samples} 个样本 ---")
        
        tasks_split = [async_split_and_detect_sentences(a.get('text', a.get('message')), **session_details) for a in batch_articles]
        results_split = await asyncio.gather(*tasks_split)
        
        tasks_overall = [async_get_overall_confidence(s, c, **session_details) for s, c in results_split]
        results_overall_confidence = await asyncio.gather(*tasks_overall)

        for j, article in enumerate(batch_articles):
            global_idx, sample_id, true_label = j + 1, article.get('id', 'N/A'), article['label']
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
                    pred_lfnd, loop_numbers, avg_confidence = lfnd_results[idx]
                    param_key = tuple(config.values())
                    if pred_lfnd == true_label: lfnd_correct_counts[param_key] += 1
                    lfnd_total_loops[param_key] += loop_numbers

                    res_lfnd = '✓ 正确' if pred_lfnd == true_label else '✗ 错误'
                    # 现在的迭代轮数会更有意义
                    print(f"    - (Sim_Thresh={config['sim_thresh']}, Logic_Thresh={config['logic_thresh']}): "
                          f"预测 {pred_lfnd:<4} -> {res_lfnd} (平均置信度: {avg_confidence:.3f}, 收敛于 {loop_numbers} 轮)")

            print(f"------------------------------------------------------------")
            clear_detection_cache()

        end_time = time.time()
        print(f"\n{'*'*30} 数据集 [{dataset_name}] 实验结果 {'*'*30}")
        print(f"总样本数: {total_samples}, 总耗时: {end_time - start_time:.2f} 秒")
        
        acc_llm = llm_correct_count / total_samples if total_samples > 0 else 0
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
                acc = lfnd_correct_counts[(s, l)] / total_samples if total_samples > 0 else 0
                cons = lfnd_total_loops[(s, l)] / total_samples if total_samples > 0 else 0
                print(f"    - (Sim_Thresh={s}, Logic_Thresh={l}): 准确率: {acc:.2%}, 平均收敛轮数: {cons:.2f} 轮")
                dataset_results[f"LFND_Sim{s}_Logic{l}"] = {"accuracy": f"{acc:.2%}", "consistency": f"{cons:.2f}"}
        all_final_results[dataset_name] = dataset_results

    print(f"\n{'='*80}\n[所有实验完成]\n{'='*80}")
    print(json.dumps(all_final_results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())