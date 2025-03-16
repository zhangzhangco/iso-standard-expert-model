#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import argparse
import requests
from tqdm import tqdm

# 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen:latest"  # 默认使用qwen模型

def load_config(config_file="config.json"):
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_sections(json_file):
    """加载章节数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_questions_count(section):
    """根据章节内容长度计算应生成的问题数量"""
    content_length = len(section["content"])
    
    # 基础问题数量
    if content_length < 500:
        question_count = 1
    elif content_length < 1000:
        question_count = 2
    elif content_length < 2000:
        question_count = 3
    elif content_length < 3000:
        question_count = 4
    else:
        question_count = 5
    
    return question_count

def parse_qa_from_text(text, section_title):
    """从文本中解析问答对"""
    # 创建debug目录（如果不存在）
    debug_dir = "debug_files"
    os.makedirs(debug_dir, exist_ok=True)
    
    # 保存原始响应到文件，用于调试
    debug_file = os.path.join(debug_dir, f"debug_{section_title.replace(' ', '_').replace('/', '_')[:30]}.txt")
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"已保存原始响应到 {debug_file}")
    
    # 尝试提取问答对
    qa_pairs = []
    
    # 使用正则表达式匹配问题和答案
    # 模式1：问题X: ... 答案X: ...
    pattern1 = r"(?:问题|问答对|Q)[\s:]*([\d]+)[\.:]?\s*(.*?)[\n\r]+(?:答案|A)[\s:]*([\d]+)[\.:]?\s*(.*?)(?=(?:\n\r*问题|问答对|Q)[\s:]*[\d]+|$)"
    matches1 = re.findall(pattern1, text, re.DOTALL)
    
    # 模式2：X. 问题: ... 答案: ...
    pattern2 = r"(?:[\d]+)[\.:]?\s*(?:问题|问答对|Q)[\.:]?\s*(.*?)[\n\r]+(?:答案|A)[\.:]?\s*(.*?)(?=(?:\n\r*[\d]+[\.:]?\s*(?:问题|问答对|Q))|$)"
    matches2 = re.findall(pattern2, text, re.DOTALL)
    
    # 模式3：问题: ... 答案: ...（没有编号）
    pattern3 = r"(?:问题|问答对|Q)[\.:]?\s*(.*?)[\n\r]+(?:答案|A)[\.:]?\s*(.*?)(?=(?:\n\r*(?:问题|问答对|Q)[\.:])|$)"
    matches3 = re.findall(pattern3, text, re.DOTALL)
    
    # 模式4：Q: ... A: ...
    pattern4 = r"Q[\.:]?\s*(.*?)[\n\r]+A[\.:]?\s*(.*?)(?=(?:\n\r*Q[\.:])|$)"
    matches4 = re.findall(pattern4, text, re.DOTALL)
    
    # 处理模式1的匹配结果
    for match in matches1:
        q_num, question, a_num, answer = match
        if q_num == a_num:  # 确保问题和答案编号一致
            qa_pairs.append({
                "instruction": question.strip(),
                "input": "",
                "output": answer.strip()
            })
    
    # 如果模式1没有匹配到，尝试模式2
    if not qa_pairs and matches2:
        for question, answer in matches2:
            qa_pairs.append({
                "instruction": question.strip(),
                "input": "",
                "output": answer.strip()
            })
    
    # 如果模式2没有匹配到，尝试模式3
    if not qa_pairs and matches3:
        for question, answer in matches3:
            qa_pairs.append({
                "instruction": question.strip(),
                "input": "",
                "output": answer.strip()
            })
    
    # 如果模式3没有匹配到，尝试模式4
    if not qa_pairs and matches4:
        for question, answer in matches4:
            qa_pairs.append({
                "instruction": question.strip(),
                "input": "",
                "output": answer.strip()
            })
    
    # 如果所有模式都没有匹配到，创建一个基于章节标题的问答对
    if not qa_pairs:
        print("警告：无法从文本中提取问答对，创建一个基于章节标题的问答对")
        fallback_qa = {
            "instruction": f"请解释ISO/IEC Directives, Part 2中关于'{section_title}'的规定和要求？",
            "input": "",
            "output": f"根据ISO/IEC Directives, Part 2的规定，关于'{section_title}'的内容如下：\n\n{text}",
            "reasoning_content_tokens_len": "",
            "content_tokens_len": "",
            "score": ""
        }
        qa_pairs.append(fallback_qa)
    
    print(f"成功解析 {len(qa_pairs)} 个问答对")
    return qa_pairs

def generate_qa_with_ollama(section):
    """使用Ollama模型生成增强的问答对"""
    # 计算应生成的问题数量
    question_count = calculate_questions_count(section)
    
    # 构建更简单的提示，不要求JSON格式输出
    prompt = f"""你是一个ISO标准化文件编写导则专家，精通ISO/IEC Directives, Part 2的所有规则和指南。

请根据以下内容，生成{question_count}个高质量的问答对。问题应该是用户可能会问的关于ISO标准文档编写的专业问题，答案必须完全基于提供的内容，并且要非常准确、专业。

章节标题：{section["title"]}
章节内容：{section["content"]}

请按以下格式输出：

问题1: [问题内容]
答案1: [答案内容]

问题2: [问题内容]
答案2: [答案内容]

...以此类推。

请确保问题具有挑战性，答案详细且准确。"""

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
        
        # 提取生成的文本
        generated_text = result.get("response", "")
        
        # 解析问答对
        qa_pairs = parse_qa_from_text(generated_text, section["title"])
        
        return qa_pairs
    except Exception as e:
        print(f"API调用错误: {e}")
        return []

def save_dataset(dataset, output_file, is_final=False):
    """保存数据集到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    if is_final:
        print(f"已保存最终数据集到 {output_file}，共 {len(dataset)} 个问答对")
    else:
        print(f"已保存中间数据集到 {output_file}，当前共 {len(dataset)} 个问答对")

def create_dataset(sections, output_file, max_sections=None, save_interval=5):
    """创建问答数据集"""
    # 创建备份文件
    if os.path.exists(output_file):
        backup_file = f"{output_file}.bak"
        print(f"备份现有数据集到 {backup_file}")
        with open(output_file, 'r', encoding='utf-8') as f_in:
            with open(backup_file, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
    
    # 加载现有数据集（如果存在）
    dataset = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"加载现有数据集，包含 {len(dataset)} 个问答对")
    
    # 限制处理的章节数量
    if max_sections and max_sections > 0:
        sections = sections[:max_sections]
        print(f"限制处理章节数量为 {max_sections}")
    
    # 创建已处理章节的集合
    processed_titles = set()
    for qa in dataset:
        if "section_title" in qa:
            processed_titles.add(qa["section_title"])
    
    # 处理每个章节
    total_sections = len(sections)
    for i, section in enumerate(sections):
        # 跳过已处理的章节
        if section["title"] in processed_titles:
            print(f"跳过已处理的章节: {section['title']}")
            continue
        
        print(f"\n处理章节 {i+1}/{total_sections}: {section['title']}")
        
        # 生成问答对
        qa_pairs = generate_qa_with_ollama(section)
        
        # 添加章节标题到每个问答对
        for qa in qa_pairs:
            qa["section_title"] = section["title"]
        
        # 添加到数据集
        dataset.extend(qa_pairs)
        
        # 定期保存数据集
        if (i + 1) % save_interval == 0 or i == total_sections - 1:
            save_dataset(dataset, output_file, is_final=(i == total_sections - 1))
        
        # 添加延迟，避免API限制
        if i < total_sections - 1:
            time.sleep(1)
    
    return dataset

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="生成ISO标准问答数据集")
    parser.add_argument("--sections_file", type=str, default="data/iso_sections.json", help="章节数据文件路径")
    parser.add_argument("--output_file", type=str, default="data/iso_alpaca_dataset.json", help="输出数据集文件路径")
    parser.add_argument("--max_sections", type=int, default=None, help="处理的最大章节数量")
    parser.add_argument("--save_interval", type=int, default=5, help="保存数据集的间隔（章节数）")
    parser.add_argument("--model", type=str, default="qwen:latest", help="使用的Ollama模型名称")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    args = parser.parse_args()
    
    # 设置全局模型名称
    global MODEL_NAME
    MODEL_NAME = args.model
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置代理（如果配置中有）
    if "proxy" in config and config["proxy"]:
        http_proxy = config["proxy"].get("http", "")
        https_proxy = config["proxy"].get("https", "")
        if http_proxy or https_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["HTTPS_PROXY"] = https_proxy
            print(f"已设置代理: HTTP={http_proxy}, HTTPS={https_proxy}")
    
    # 加载章节数据
    print(f"从 {args.sections_file} 加载章节数据...")
    sections = load_sections(args.sections_file)
    print(f"加载了 {len(sections)} 个章节")
    
    # 创建数据集
    dataset = create_dataset(sections, args.output_file, args.max_sections, args.save_interval)
    
    print(f"数据集生成完成，共 {len(dataset)} 个问答对")

if __name__ == "__main__":
    main() 