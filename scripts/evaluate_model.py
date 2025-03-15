#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="评估微调后的模型")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/deepseek-llm-7b-base", help="基础模型名称")
    parser.add_argument("--peft_model", type=str, default="./models", help="PEFT模型路径")
    parser.add_argument("--test_file", type=str, default=None, help="测试数据集路径，如果不提供则使用训练数据的一部分")
    parser.add_argument("--train_file", type=str, default="data/iso_alpaca_dataset.json", help="训练数据集路径")
    parser.add_argument("--num_test_samples", type=int, default=20, help="测试样本数量")
    parser.add_argument("--output_file", type=str, default="models/evaluation_results.json", help="评估结果输出文件")
    parser.add_argument("--use_4bit", action="store_true", help="是否使用4位量化")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="生成的最大新token数")
    
    return parser.parse_args()

def format_prompt(instruction, input_text=""):
    """格式化提示文本"""
    if input_text.strip():
        prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 响应:\n"
    else:
        prompt = f"### 指令:\n{instruction}\n\n### 响应:\n"
    
    return prompt

def load_model(args):
    """加载模型和分词器"""
    print(f"加载基础模型: {args.base_model}")
    
    # 设置量化配置
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        padding_side="right",
        trust_remote_code=True
    )
    
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"加载PEFT模型: {args.peft_model}")
    # 加载PEFT模型
    model = PeftModel.from_pretrained(base_model, args.peft_model)
    
    return model, tokenizer

def get_test_samples(args):
    """获取测试样本"""
    if args.test_file and os.path.exists(args.test_file):
        print(f"从测试文件加载样本: {args.test_file}")
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        if len(test_data) > args.num_test_samples:
            test_samples = random.sample(test_data, args.num_test_samples)
        else:
            test_samples = test_data
    else:
        print(f"从训练文件中抽取测试样本: {args.train_file}")
        with open(args.train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            
        # 随机抽取测试样本
        test_samples = random.sample(train_data, min(args.num_test_samples, len(train_data)))
    
    print(f"选择了 {len(test_samples)} 个测试样本")
    return test_samples

def generate_responses(model, tokenizer, test_samples, args):
    """生成响应"""
    print("生成响应...")
    results = []
    
    model.eval()
    with torch.no_grad():
        for sample in tqdm(test_samples):
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            expected_output = sample["output"]
            
            # 准备提示
            prompt = format_prompt(instruction, input_text)
            
            # 对提示进行标记
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 生成响应
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 解码响应
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的响应部分
            response = generated_text[len(prompt):]
            
            # 添加到结果
            results.append({
                "instruction": instruction,
                "input": input_text,
                "expected": expected_output,
                "generated": response
            })
    
    return results

def main():
    args = parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args)
    
    # 获取测试样本
    test_samples = get_test_samples(args)
    
    # 生成响应
    results = generate_responses(model, tokenizer, test_samples, args)
    
    # 保存结果
    print(f"保存评估结果到: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("评估完成!")

if __name__ == "__main__":
    main() 