#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 支持的模型列表
SUPPORTED_MODELS = [
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/deepseek-r1-distill-qwen-7b",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-1.5B"
]

def load_config(config_file="config.json"):
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def prepare_dataset(dataset_path, tokenizer, max_length=2048):
    """准备数据集"""
    # 加载数据集
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    # 根据模型类型选择不同的提示模板
    if "deepseek" in tokenizer.name_or_path.lower():
        # DeepSeek模型的提示模板
        prompt_template = """<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""
    elif "qwen" in tokenizer.name_or_path.lower():
        # Qwen模型的提示模板
        prompt_template = """<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""
    else:
        # 默认提示模板
        prompt_template = """### 问题：
{instruction}

### 回答：
{output}"""
    
    def format_prompt(example):
        # 格式化提示
        prompt = prompt_template.format(
            instruction=example["instruction"],
            output=example["output"]
        )
        return {"text": prompt}
    
    # 应用格式化
    formatted_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    # 对数据集进行分词
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def train(args):
    """使用LoRA微调模型"""
    print(f"正在使用LoRA微调模型: {args.model_name}")
    
    # 加载配置
    config = load_config(args.config_file)
    
    # 设置Hugging Face API密钥环境变量
    hf_api_key = config.get("huggingface", {}).get("api_key")
    if hf_api_key:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_api_key
        print(f"已从配置文件设置Hugging Face API密钥")
    
    # 设置代理环境变量（如果配置中有）
    proxy_settings = config.get("proxy", {})
    http_proxy = proxy_settings.get("http")
    https_proxy = proxy_settings.get("https")
    
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
        print(f"已设置HTTP代理: {http_proxy}")
    
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
        print(f"已设置HTTPS代理: {https_proxy}")
    
    # 检查模型是否支持
    if args.model_name not in SUPPORTED_MODELS:
        print(f"警告: 模型 {args.model_name} 不在官方支持列表中，可能需要额外调整")
    
    # 设置量化配置
    if args.use_4bit:
        print("使用4位量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        print("使用8位量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # 加载模型和分词器
    print(f"加载模型: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    # 确保分词器有正确的特殊标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备模型进行训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 适用于大多数Transformer模型
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% 的总参数)")
    
    # 准备数据集
    train_dataset = prepare_dataset(args.dataset_path, tokenizer, args.max_length)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        save_strategy="epoch",
        logging_steps=10,
        fp16=not args.bf16,
        bf16=args.bf16,
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"lora-{args.model_name.split('/')[-1]}" if args.use_wandb else None,
    )
    
    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print(f"保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("训练完成!")

def main():
    parser = argparse.ArgumentParser(description="使用LoRA微调大语言模型")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-r1-distill-qwen-7b", help="模型名称")
    parser.add_argument("--dataset_path", type=str, default="data/iso_alpaca_dataset.json", help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="models/lora", help="输出目录")
    parser.add_argument("--use_4bit", action="store_true", help="使用4位量化")
    parser.add_argument("--bf16", action="store_true", help="使用bf16精度")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="学习率调度器")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--use_wandb", action="store_true", help="使用Weights & Biases记录训练")
    parser.add_argument("--config_file", type=str, default="config.json", help="配置文件路径")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main() 