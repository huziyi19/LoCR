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