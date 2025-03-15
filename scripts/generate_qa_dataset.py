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

def calculate_questions_count(section):
    """根据章节内容动态计算应生成的问题数量"""
    content = section["content"]
    title = section["title"]
    
    # 基础问题数量
    base_count = 5
    
    # 根据内容长度增加问题数量
    # 每1000个字符增加1个问题，最多增加5个问题
    length_factor = min(5, len(content) // 1000)
    
    # 根据是否包含关键词增加问题数量
    keywords = ["shall", "should", "may", "can", "requirement", "recommendation", 
               "permission", "possibility", "normative", "informative",
               "technical", "specification", "procedure", "guide", "mandatory"]
    
    keyword_count = sum(1 for keyword in keywords if keyword in content.lower() or keyword in title.lower())
    keyword_factor = min(3, keyword_count // 3)  # 每3个关键词增加1个问题，最多增加3个问题
    
    # 计算最终问题数量，确保不少于5个
    question_count = max(5, base_count + length_factor + keyword_factor)
    
    return question_count

def generate_qa_with_ollama(section):
    """使用Ollama模型生成增强的问答对"""
    # 计算应生成的问题数量
    question_count = calculate_questions_count(section)
    
    # 构建更专业的提示
    prompt = f"""你是一个ISO标准化文件编写导则专家，精通ISO/IEC Directives, Part 2的所有规则和指南。
根据以下内容，生成{question_count}个高质量的问答对。问题应该是用户可能会问的关于ISO标准文档编写的专业问题，答案必须完全基于提供的内容，并且要非常准确、专业。

章节标题：{section["title"]}
章节内容：{section["content"]}

请特别关注以下几点：
1. 确保准确解释ISO标准中的专业术语（如"shall"、"should"、"may"等）
2. 包含具体的规则编号和引用
3. 提供实际的例子来说明规则的应用
4. 解释规则背后的原因和重要性

内容重要性评估：本段落包含{question_count}个值得提问的知识点。请确保充分覆盖这些知识点。

为每个问答对生成一个JSON对象，包含以下字段：
1. "instruction": 用户的问题，以问号结尾，问题应该具体且专业
2. "input": 空字符串 
3. "output": 首先包含一个<think>标签内的思考过程，然后是根据章节内容给出的详细专业回答

格式示例：
{{
  "instruction": "ISO标准中'shall'和'should'有什么区别？",
  "input": "",
  "output": "<think>这个问题涉及ISO标准中的关键用词。根据ISO/IEC Directives, Part 2的规定，'shall'和'should'有明确的不同含义和用法。我需要准确解释这些区别，并提供具体的例子。</think> 根据ISO/IEC Directives, Part 2第7.3.2节，'shall'和'should'在ISO标准中有明确的区别：\\n\\n1. 'shall'：表示严格的要求，必须遵守才能符合标准。它用于表达要求。例如：'The test report shall include the following information.'（测试报告应包含以下信息。）\\n\\n2. 'should'：表示在几种可能性中特别推荐的一种，但不是强制性的。它用于表达建议。例如：'Test reports should be delivered within 5 working days.'（测试报告应该在5个工作日内交付。）\\n\\n这种区别对于标准的实施和合规性评估非常重要。使用'shall'的条款是强制性的，而使用'should'的条款则是推荐性的。"
}}

所有问答对应该合并为一个JSON数组。仅返回JSON格式，不要有任何其他解释。
"""

    # 打印生成信息
    print(f"为'{section['title']}'生成{question_count}个问题...")

    # 调用Ollama API
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5,  # 降低温度以获得更确定性的回答
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
                
                print(f"成功生成 {len(qa_pairs)} 个问答对")
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

def create_dataset(sections, output_file, max_sections=None, save_interval=5):
    """创建Alpaca格式的问答数据集"""
    dataset = []
    
    # 如果指定了最大章节数，优先选择包含关键词的章节
    if max_sections and max_sections < len(sections):
        # 优先选择包含关键词的章节
        key_sections = []
        other_sections = []
        
        keywords = ["shall", "should", "may", "can", "requirement", "recommendation", 
                   "permission", "possibility", "normative", "informative"]
        
        for section in sections:
            content = section["content"].lower()
            title = section["title"].lower()
            
            if any(keyword in content or keyword in title for keyword in keywords):
                key_sections.append(section)
            else:
                other_sections.append(section)
        
        # 为关键章节按内容长度排序，优先选择较长的章节
        key_sections.sort(key=lambda x: len(x["content"]), reverse=True)
        
        # 确保关键章节被选中
        if len(key_sections) <= max_sections:
            selected_sections = key_sections
            remaining = max_sections - len(key_sections)
            if remaining > 0 and other_sections:
                # 其他章节也按长度排序
                other_sections.sort(key=lambda x: len(x["content"]), reverse=True)
                selected_sections.extend(other_sections[:remaining])
        else:
            selected_sections = key_sections[:max_sections]
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
        time.sleep(1)
    
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
    parser.add_argument("--save_interval", type=int, default=5, help="保存中间结果的间隔")
    parser.add_argument("--output", type=str, default="data/iso_alpaca_dataset.json", help="输出文件路径")
    parser.add_argument("--min_questions", type=int, default=5, help="每个章节最少生成的问题数量")
    args = parser.parse_args()
    
    input_file = "data/iso_sections.json"
    output_file = args.output
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return
    
    sections = load_sections(input_file)
    print(f"加载了 {len(sections)} 个章节")
    
    create_dataset(sections, output_file, args.max_sections, args.save_interval)
    
    print("处理完成!")

if __name__ == "__main__":
    main() 