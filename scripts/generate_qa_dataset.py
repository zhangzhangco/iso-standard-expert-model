#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import random
import os
import requests
import argparse
from tqdm import tqdm

# Ollama API配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwq:latest"  # 使用本地ollama部署的qwq模型

def load_sections(json_file):
    """加载章节JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        sections = json.load(f)
    return sections

def generate_qa_with_ollama(section):
    """使用Ollama模型生成问答对"""
    # 构建提示
    prompt = f"""你是一个ISO标准化文件编写导则专家。
根据以下内容，生成3个高质量的问答对。问题应该是用户可能会问的关于ISO标准文档编写的问题，答案应该完全基于提供的内容。

章节标题：{section["title"]}
章节内容：{section["content"]}

为每个问答对生成一个JSON对象，包含以下字段：
1. "instruction": 用户的问题，以问号结尾
2. "input": 空字符串 
3. "output": 首先包含一个<think>标签内的思考过程，然后是根据章节内容给出的详细回答

格式示例：
{{
  "instruction": "问题？",
  "input": "",
  "output": "<think>这里是分析问题和组织答案的思考过程</think> 这里是最终给用户的回答。"
}}

所有问答对应该合并为一个JSON数组。仅返回JSON格式，不要有任何其他解释。
"""

    # 调用Ollama API
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 64000
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # 尝试解析返回的JSON
        try:
            # 提取生成的文本
            generated_text = result.get("response", "")
            
            # 找到JSON部分
            json_start = generated_text.find("[")
            json_end = generated_text.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = generated_text[json_start:json_end]
                qa_pairs = json.loads(json_text)
                
                # 添加额外的空字段以匹配示例格式
                for qa in qa_pairs:
                    qa["repo_name"] = ""
                    qa["prompt_tokens_len"] = ""
                    qa["reasoning_content_tokens_len"] = ""
                    qa["content_tokens_len"] = ""
                    qa["score"] = ""
                
                return qa_pairs
            else:
                print(f"无法从响应中提取JSON: {generated_text}")
                return []
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {generated_text}")
            return []
    except Exception as e:
        print(f"API调用错误: {e}")
        return []

def save_dataset(dataset, output_file, is_final=False):
    """保存数据集到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    if is_final:
        print(f"已完成所有生成，总共 {len(dataset)} 个问答对，保存到 {output_file}")
    else:
        print(f"中间保存：当前已生成 {len(dataset)} 个问答对")

def create_dataset(sections, output_file, max_sections=None, save_interval=10):
    """创建Alpaca格式的问答数据集"""
    dataset = []
    
    # 如果指定了最大章节数，随机选择章节
    if max_sections and max_sections < len(sections):
        selected_sections = random.sample(sections, max_sections)
    else:
        selected_sections = sections
    
    # 尝试加载已有的数据集（如果存在）
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"加载已有数据集，包含 {len(dataset)} 个问答对")
        except:
            print("无法加载已有数据集，将创建新的数据集")
    
    # 创建一个临时文件名，用于中间结果保存
    temp_output_file = output_file + ".temp"
    
    # 为每个章节生成问答对
    for i, section in enumerate(tqdm(selected_sections, desc="生成问答对")):
        # 生成问答对
        qa_pairs = generate_qa_with_ollama(section)
        
        if qa_pairs:
            # 直接添加生成的问答对到数据集
            dataset.extend(qa_pairs)
        
        # 定期保存中间结果
        if (i + 1) % save_interval == 0:
            save_dataset(dataset, temp_output_file)
        
        # 避免API限制，添加短暂延迟
        time.sleep(0.5)
    
    # 保存最终数据集
    save_dataset(dataset, output_file, is_final=True)
    
    # 删除临时文件
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)
    
    return dataset

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="生成ISO标准问答数据集")
    parser.add_argument("--max_sections", type=int, default=100, help="处理的最大章节数")
    parser.add_argument("--save_interval", type=int, default=1, help="保存中间结果的间隔")
    args = parser.parse_args()
    
    input_file = "data/iso_sections.json"
    output_file = "data/iso_alpaca_dataset.json"
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return
    
    sections = load_sections(input_file)
    print(f"加载了 {len(sections)} 个章节")
    
    create_dataset(sections, output_file, args.max_sections, args.save_interval)
    
    print("处理完成!")

if __name__ == "__main__":
    main() 