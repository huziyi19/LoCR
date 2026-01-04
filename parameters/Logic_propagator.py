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