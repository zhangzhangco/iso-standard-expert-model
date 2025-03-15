#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

def load_model(base_model_name="deepseek-ai/deepseek-r1-distill-qwen-7b", 
               peft_model_path="./models/fixed", 
               use_4bit=True):
    """加载模型和分词器"""
    print(f"加载基础模型: {base_model_name}")
    
    # 设置量化配置
    model_kwargs = {}
    if use_4bit:
        print("使用4位量化...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True
    )
    
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # 加载简化的配置
    print(f"加载PEFT配置: {peft_model_path}")
    config_path = os.path.join(peft_model_path, "simplified_adapter_config.json")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 创建LoraConfig对象
    lora_config = LoraConfig(
        r=config_dict["r"],
        lora_alpha=config_dict["lora_alpha"],
        target_modules=config_dict["target_modules"],
        lora_dropout=config_dict["lora_dropout"],
        bias=config_dict["bias"],
        task_type=config_dict["task_type"]
    )
    
    # 创建PEFT模型
    print("创建PEFT模型...")
    model = get_peft_model(base_model, lora_config)
    
    # 加载权重
    print("加载权重...")
    model.load_adapter(peft_model_path, adapter_name="default")
    
    print("模型加载成功！")
    return model, tokenizer

def format_prompt(instruction):
    """格式化提示文本"""
    # 添加专业上下文
    context = "你是一位ISO标准文档编写专家，精通ISO/IEC Directives, Part 2的所有规则和指南。请提供准确、专业的回答，引用相关章节和规则。请特别注意'shall'、'should'、'may'等关键词的正确用法。"
    # DeepSeek模型的提示模板
    prompt = f"<|im_start|>user\n{context}\n\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
    return prompt

def generate_response(model, tokenizer, instruction, 
                     max_new_tokens=2048, 
                     temperature=0.6, 
                     top_p=0.95,
                     repetition_penalty=1.1):
    """生成响应"""
    # 准备提示
    prompt = format_prompt(instruction)
    
    # 对提示进行标记
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成响应
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 解码响应
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取生成的响应部分
    response_start = generated_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    response_end = generated_text.find("<|im_end|>", response_start)
    
    if response_end == -1:  # 如果没有找到结束标记
        response = generated_text[response_start:]
    else:
        response = generated_text[response_start:response_end]
    
    return response.strip()

def interactive_mode(model, tokenizer):
    """交互模式"""
    print("\n欢迎使用ISO标准专家模型！")
    print("输入'退出'或'exit'结束对话\n")
    
    while True:
        user_input = input("\n请输入您的问题: ")
        if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
            print("谢谢使用，再见！")
            break
        
        print("\n正在生成回答...")
        response = generate_response(model, tokenizer, user_input)
        print(f"\n回答: {response}")

def main():
    parser = argparse.ArgumentParser(description="ISO标准专家模型查询工具")
    parser.add_argument("--query", type=str, help="单次查询的问题")
    parser.add_argument("--interactive", action="store_true", help="启用交互模式")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/deepseek-r1-distill-qwen-7b", help="基础模型名称")
    parser.add_argument("--peft_model", type=str, default="./models/fixed", help="PEFT模型路径")
    parser.add_argument("--max_tokens", type=int, default=2048, help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.3, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p采样参数")
    parser.add_argument("--no_4bit", action="store_true", help="不使用4位量化")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(
        base_model_name=args.base_model,
        peft_model_path=args.peft_model,
        use_4bit=not args.no_4bit
    )
    
    # 根据参数选择模式
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.query:
        print("\n正在生成回答...")
        response = generate_response(
            model, 
            tokenizer, 
            args.query,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"\n问题: {args.query}")
        print(f"\n回答: {response}")
    else:
        print("请提供查询问题(--query)或启用交互模式(--interactive)")

if __name__ == "__main__":
    main() 