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