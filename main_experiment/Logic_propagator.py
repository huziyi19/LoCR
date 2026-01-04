import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline
from itertools import combinations
import torch

##jia

import re
import difflib
import numpy as np


NEGATION_WORDS = {"not", "no", "never", "n't", "didn't", "doesn't", "don't", "cannot", "won't", "wasn't", "isn't"}
##jia

# 加载自然语言推理模型
nli_pipeline = pipeline("text-classification", model="roberta-large-mnli")

'''
##jia
# 使用轻量模型，也可改为 'all-mpnet-base-v2' 提高准确率
model = SentenceTransformer('all-MiniLM-L6-v2')


# 常见否定词，用于检测not关系
NEGATION_WORDS = {"not", "n't", "never", "no", "none", "nothing", "nowhere", "neither", "didn't", "cannot", "without"}

def contains_negation(text):
    """判断句子是否包含否定词"""
    return any(neg in text.lower() for neg in NEGATION_WORDS)

def is_not_relation(sent1, sent2, threshold=0.5):
    """判断两个句子是否为相反关系（not）"""
    neg1 = contains_negation(sent1)
    neg2 = contains_negation(sent2)

    if neg1 != neg2:
        emb1 = model.encode(sent1, convert_to_tensor=True)
        emb2 = model.encode(sent2, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2).item()
        if sim < threshold:
            return True
    return False

NEGATION_WORDS = {"not", "no", "never", "n't", "didn't", "doesn't", "don't", "cannot", "won't", "wasn't", "isn't", "nothing", "none", "nobody", "nowhere", "neither", "without"}

def is_negation(sent1, sent2):
    """
    判断两个句子是否互为否定
    基于否定词检测 + 高语义相似度
    """

    def contains_negation(sentence):
        return any(neg in sentence.lower() for neg in NEGATION_WORDS)

    has_neg1 = contains_negation(sent1)
    has_neg2 = contains_negation(sent2)

    # 一个含否定词，一个不含，且语义相反
    if has_neg1 != has_neg2:
        emb1 = model.encode(sent1, convert_to_tensor=True)
        emb2 = model.encode(sent2, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2).item()
        if sim > 0.6:
            return True
    return False

def find_char_level_contained_pairs(sent1,sent2,max_char_diff=1):

    def clean_and_remove_negation(sentence):
        """
        统一小写、去标点、去否定词
        """
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)  # 去掉标点
        words = sentence.split()
        filtered = [word for word in words if word not in NEGATION_WORDS]
        return ' '.join(filtered)

    def char_diff(s1, s2):
        """
        返回 s1 中有多少字符不在 s2 中
        """
        s1_chars = list(s1.replace(" ", ""))
        s2_chars = list(s2.replace(" ", ""))
        diff = [c for c in s1_chars if c not in s2_chars]
        return len(diff)

    def is_match_on_char_level(short, long, max_char_diff=1):
        """
        判断短句是否基本被长句“字符级包含”，允许 max_char_diff 个字母差异
        """
        short_clean = clean_and_remove_negation(short)
        long_clean = clean_and_remove_negation(long)

        if len(short_clean) >= len(long_clean):
            return False

        diff = char_diff(short_clean, long_clean)
        return diff <= max_char_diff

    pairs_label=is_match_on_char_level(sent1, sent2, max_char_diff=max_char_diff)
    return pairs_label


'''
def tokenize(sentence):
    """小写、去标点、分词"""
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)  # 去除标点
    return sentence.split()

def is_logical_negation(sent1, sent2, min_similarity=0.6):
    """
    判断 sent1 和 sent2 是否互为“非”
    - 结构相似度高
    - 一个包含否定词，另一个不包含
    - 去掉否定词后基本相同
    """
    words1 = tokenize(sent1)
    words2 = tokenize(sent2)

    # 判断否定分布情况
    neg1 = any(w in NEGATION_WORDS for w in words1)
    neg2 = any(w in NEGATION_WORDS for w in words2)

    # 必须有一个含否定，另一个不含
    if neg1 == neg2:
        return False

    # 去掉否定词
    clean1 = [w for w in words1 if w not in NEGATION_WORDS]
    clean2 = [w for w in words2 if w not in NEGATION_WORDS]

    # 字面字符串形式匹配（更敏感）
    str1 = ' '.join(clean1)
    str2 = ' '.join(clean2)

    similarity = difflib.SequenceMatcher(None, str1, str2).ratio()

    return similarity >= min_similarity


##jia


def extract_logical_edges(nodes):
    edges = []

    #index_to_label = {i: data['label'] for i, data in nodes}

    # 比较所有句子对
    for (i1, label1), (i2, label2) in combinations(nodes, 2):
            # Perform NLI prediction
            # The pipeline will handle truncation
        result1 = nli_pipeline((label1['label'], label2['label']), padding=True, truncation=True)
        score1 = result1['score'] if result1['label'] == 'ENTAILMENT' else 1 - result1['score']

        result2 = nli_pipeline((label2['label'], label1['label']), padding=True, truncation=True)
        score2 = result2['score'] if result2['label'] == 'ENTAILMENT' else 1 - result2['score']

        ''' 
        # print(label1,label2)
        # print('1+1')
        '''

        # 处理 result1 的逻辑关系
        if result1['label'] == 'ENTAILMENT' and result1['score'] > 0.8:
            edges.append((i1, i2, {'label': 'imply'}))
        elif result1['label'] == 'CONTRADICTION' and result1['score'] > 0.8:
            if is_logical_negation(label1['label'], label2['label']):
            #if is_negation(label1['label'], label2['label']) or find_char_level_contained_pairs(label1['label'], label2['label'],max_char_diff=1):
                edges.append((i1, i2, {'label': 'not'}))

        # 处理 result2 的逻辑关系
        if result2['label'] == 'ENTAILMENT' and result2['score'] > 0.8:
            edges.append((i2, i1, {'label': 'imply'}))
        elif result2['label'] == 'CONTRADICTION' and result2['score'] > 0.8:
            if is_logical_negation(label1['label'], label2['label']):
            #if is_negation(label1['label'], label2['label']) or find_char_level_contained_pairs(label1['label'], label2['label'],max_char_diff=1):
                edges.append((i2, i1, {'label': 'not'}))
        '''

        # 如果一方蕴含另一方（反方向），也可以推断‘被蕴含’
        if result1['label'] == 'ENTAILMENT' and result1['score'] > 0.8 and \
                result2['label'] != 'ENTAILMENT':
            edges.append((i2, i1, {'label': 'implied'}))
        elif result2['label'] == 'ENTAILMENT' and result2['score'] > 0.8 and \
                result1['label'] != 'ENTAILMENT':
            edges.append((i1, i2, {'label': 'implied'}))
        '''
    return edges



def logic_propagation(nodes,feature_matrix):
    '''
    def logic_matrix(node_set):
        n = len(node_set)
        L_matrix = np.zeros((n, n))
        edges = extract_logical_edges(node_set)
        for i, j, logic in edges:
            if logic['label'] == 'not':
                L_matrix[i - 1, j - 1] = 1
                L_matrix[j - 1, i - 1] = 1
            if logic['label'] == 'imply':
                L_matrix[i - 1, j - 1] = 2
        return L_matrix

    L=logic_matrix(nodes)
    '''

    n = len(nodes)
    #f_matrix = np.zeros((n,1))
    f_matrix = feature_matrix.copy()

    edges = extract_logical_edges(nodes)

    '''
    #edges = [(0,1,{'label':'not'}),(1,0,{'label':'not'}),(5,2,{'label':'imply'})]
    #print('edges____',edges)
    #bug_point
    '''

    for i, j, logic in edges:

        '''
        调试
        print('logic____',logic)
        print('pre_i____',feature_matrix[i,0])
        print('pre_j____', feature_matrix[j, 0])
        '''

        if logic['label'] == 'not':
            if np.isnan(feature_matrix[i,0]) and (not np.isnan(feature_matrix[j,0])):
                f_matrix[i,0] = 1 - feature_matrix[j,0]
                f_matrix[j,0] = feature_matrix[j,0]
            elif np.isnan(feature_matrix[j,0]) and (not np.isnan(feature_matrix[i,0])):
                f_matrix[j,0] = 1 - feature_matrix[i,0]
                f_matrix[i,0] = feature_matrix[i,0]
        if logic['label'] == 'imply':

            '''
            调试
            #print(np.isnan(feature_matrix[i - 1,0]))
            #print(np.isnan(feature_matrix[j - 1,0]))
            #print('feature_____',feature_matrix[i - 1,0])
            '''

            if (not np.isnan(feature_matrix[i,0])) and np.isnan(feature_matrix[j,0]) :
                f_matrix[j,0] =  feature_matrix[i,0]
                f_matrix[i,0] = feature_matrix[i,0]
        '''
        调试
        print('final_i____', f_matrix[i, 0])
        print('final_j____', f_matrix[j, 0])
        '''
    return f_matrix












'''
nodes = [
    (0, {'label': 'Her father did not give her a gift.'}),
    (1, {'label': 'Her father gave her a gift.'}),
    (2, {'label': 'She got 100 and her father gave her a gift.'}),
    (3, {'label': 'She got 100.'}),
    (4, {'label': 'She failed the exam.'}),
    (5, {'label': 'She did not get 100.'}),
    (6, {'label': 'Another, favored by the pregnant.'}),
    (7, {'label': 'And non-pregnant alike.'}),
    (8, {'label': 'Non-pregnant alike.'}),
    (9, {'label': 'French fries dipped into a milkshake.'}),
    (10, {'label': 'OMG I have the best husband, Prinsloo praised her love Adam Levine in the text written over the photo, showing off her spoils from Fatburger.'}),
    (11, {'label': 'Want all the latest pregnancy.'}),
    (12, {'label': 'And birth announcements, plus celebrity mom blogs.'}),
    (13, {'label': 'Birth announcements, plus celebrity mom blogs.'}),
    (14, {'label': 'Click here to get those.'}),
    (15, {'label': 'And more in the PEOPLE Babies newsletter.'})
]

edges = extract_logical_edges(nodes)

for edge in edges:
    print(edge)
'''

