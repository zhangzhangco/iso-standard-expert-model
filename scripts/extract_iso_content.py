#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from bs4 import BeautifulSoup
import os

def extract_content_from_html(html_file):
    """从HTML文件中提取有用内容，并按照标题组织段落"""
    print(f"正在处理文件: {html_file}")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 移除脚本和样式元素
    for script in soup(["script", "style"]):
        script.extract()
    
    # 提取正文内容
    content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'li'])
    
    # 组织内容为按标题划分的段落集合
    sections = []
    current_section = None
    current_title = None
    buffer = []
    
    # 遍历所有元素，按标题组织内容
    for element in content_elements:
        text = element.get_text(strip=True)
        if not text:
            continue
            
        # 跳过版权信息和无关内容
        if any(skip_term in text.lower() for skip_term in ["copyright", "forced page break", "tips for a better"]):
            continue
        
        # 检查是否是标题元素
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # 处理前一个章节
            if current_title and buffer:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(buffer),
                    "heading_level": current_section
                })
            
            # 开始新章节
            current_title = text
            current_section = element.name
            buffer = []
        else:
            # 添加内容到当前章节
            if current_title is not None:  # 确保我们有一个当前章节
                buffer.append(text)
    
    # 添加最后一个章节
    if current_title and buffer:
        sections.append({
            "title": current_title,
            "content": "\n".join(buffer),
            "heading_level": current_section
        })
    
    # 过滤掉太短的内容
    filtered_sections = []
    for section in sections:
        if len(section["content"]) >= 50:  # 只保留内容足够长的章节
            filtered_sections.append(section)
    
    print(f"提取了 {len(filtered_sections)} 个章节")
    return filtered_sections

def save_to_json(sections, output_file):
    """将章节保存为JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    print(f"已保存到 {output_file}")

def main():
    html_file = "data/iso_directives.html"
    output_file = "data/iso_sections.json"
    
    if not os.path.exists(html_file):
        print(f"错误: 找不到文件 {html_file}")
        return
    
    sections = extract_content_from_html(html_file)
    save_to_json(sections, output_file)
    
    print("处理完成!")

if __name__ == "__main__":
    main() 