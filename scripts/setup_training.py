#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
import json

def run_command(command):
    """运行shell命令并打印输出"""
    print(f"执行: {command}")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    if process.returncode != 0:
        print(f"命令执行失败，返回码: {process.returncode}")
        return False
    return True

def load_config(config_file="config.json"):
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def setup_environment(args):
    """设置训练环境"""
    print("正在设置训练环境...")
    
    # 检查是否有GPU
    run_command("nvidia-smi")
    
    # 安装必要的依赖
    dependencies = [
        "torch>=2.0.0", 
        "transformers>=4.40.0", 
        "datasets", 
        "accelerate", 
        "peft>=0.6.0", 
        "bitsandbytes>=0.41.1", 
        "trl>=0.7.4", 
        "deepspeed",
        "wandb",
        "sentencepiece",
        "huggingface_hub"
    ]
    
    print("安装依赖...")
    # 使用一次性安装所有依赖，避免生成垃圾文件
    deps_str = " ".join(dependencies)
    run_command(f"pip install --no-cache-dir {deps_str}")
    
    # 加载配置
    config = load_config(args.config_file)
    
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
    
    print("设置Hugging Face凭证...")
    # 检查是否有API密钥
    api_key = config.get("huggingface", {}).get("api_key")
    if api_key:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = api_key
        print("使用配置文件中的Hugging Face API密钥")
        
        # 使用API密钥登录
        run_command("huggingface-cli login --token " + api_key)
    else:
        # 提示用户手动登录
        print("未找到Hugging Face API密钥，请手动登录")
        run_command("huggingface-cli login")
    
    print("环境设置完成！")

def download_model(model_name="deepseek-ai/deepseek-llm-7b-base", output_dir="models"):
    """下载预训练模型"""
    print(f"正在下载 {model_name}...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用Hugging Face CLI下载模型
    run_command(f"huggingface-cli download {model_name} --local-dir {output_dir}/{model_name.split('/')[-1]}")
    
    print(f"模型下载完成: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="设置DeepSeek模型微调环境")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-llm-7b-base", help="要下载的模型名称")
    parser.add_argument("--output_dir", type=str, default="models", help="模型保存目录")
    parser.add_argument("--config_file", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--skip_download", action="store_true", help="跳过模型下载步骤")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment(args)
    
    # 下载模型
    if not args.skip_download:
        download_model(args.model, args.output_dir)
        print("一切就绪，可以开始微调!")
    else:
        print("跳过模型下载，假设模型已存在")
        print("一切就绪，可以开始微调!")

if __name__ == "__main__":
    main() 