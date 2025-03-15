#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="将微调后的模型导出为GGUF格式")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/deepseek-llm-7b-base", help="基础模型名称")
    parser.add_argument("--peft_model", type=str, default="./models", help="PEFT模型路径")
    parser.add_argument("--output_dir", type=str, default="./models/exported_model", help="导出目录")
    parser.add_argument("--merge_lora", action="store_true", help="是否合并LoRA权重")
    parser.add_argument("--quantize", default="q4_k_m", help="量化类型: q4_0, q4_1, q5_0, q5_1, q8_0, q4_k_m（默认，平衡性能和内存）, q5_k_m, q6_k")
    
    return parser.parse_args()

def merge_lora_to_base_model(args):
    """将LoRA权重合并到基础模型"""
    print("合并LoRA权重到基础模型...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # 加载基础模型
    print(f"加载基础模型: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载和合并LoRA模型
    print(f"加载PEFT模型: {args.peft_model}")
    model = PeftModel.from_pretrained(base_model, args.peft_model)
    
    print("合并权重...")
    model = model.merge_and_unload()
    
    # 创建输出目录
    merged_model_path = os.path.join(args.output_dir, "merged_model")
    os.makedirs(merged_model_path, exist_ok=True)
    
    # 保存合并后的模型
    print(f"保存合并后的模型到: {merged_model_path}")
    model.save_pretrained(merged_model_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_path)
    
    return merged_model_path

def install_llama_cpp_python():
    """安装llama.cpp Python绑定"""
    print("安装llama.cpp Python绑定...")
    # 使用--no-cache-dir参数避免生成垃圾文件
    subprocess.run("pip install --no-cache-dir llama-cpp-python", shell=True, check=True)

def convert_to_gguf(model_path, args):
    """将模型转换为GGUF格式"""
    print("将模型转换为GGUF格式...")
    
    # 创建输出目录
    gguf_output_dir = os.path.join(args.output_dir, "gguf")
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    # 下载并使用最新的转换脚本
    if not os.path.exists("llama.cpp"):
        print("克隆llama.cpp仓库...")
        subprocess.run("git clone https://github.com/ggerganov/llama.cpp.git", shell=True, check=True)
    else:
        print("更新llama.cpp仓库...")
        subprocess.run("cd llama.cpp && git pull", shell=True, check=True)
    
    # 构建llama.cpp
    print("编译llama.cpp...")
    subprocess.run("cd llama.cpp && make", shell=True, check=True)
    
    # 运行转换脚本
    model_name = os.path.basename(args.base_model.split('/')[-1]) + "-iso-expert"
    gguf_model_path = os.path.join(gguf_output_dir, f"{model_name}.gguf")
    
    print(f"转换模型到GGUF: {gguf_model_path}")
    convert_command = f"python llama.cpp/convert.py {model_path} --outfile {gguf_model_path} --outtype f16"
    subprocess.run(convert_command, shell=True, check=True)
    
    # 量化模型
    if args.quantize:
        print(f"量化模型为 {args.quantize}...")
        quant_model_path = os.path.join(gguf_output_dir, f"{model_name}-{args.quantize}.gguf")
        quant_command = f"./llama.cpp/quantize {gguf_model_path} {quant_model_path} {args.quantize}"
        subprocess.run(quant_command, shell=True, check=True)
        gguf_model_path = quant_model_path
    
    print(f"GGUF模型已保存到: {gguf_model_path}")
    return gguf_model_path

def create_mobile_package(gguf_model_path, args):
    """创建移动设备可用的模型包"""
    print("创建移动设备可用的模型包...")
    
    mobile_output_dir = os.path.join(args.output_dir, "mobile")
    os.makedirs(mobile_output_dir, exist_ok=True)
    
    # 拷贝GGUF模型
    model_filename = os.path.basename(gguf_model_path)
    mobile_model_path = os.path.join(mobile_output_dir, model_filename)
    shutil.copy2(gguf_model_path, mobile_model_path)
    
    # 创建配置文件
    config = {
        "name": "ISO标准专家",
        "description": "ISO标准化文件编写导则专家，基于DeepSeek 7B模型微调",
        "modelType": "llama",
        "contextSize": 4096,
        "promptTemplate": "### 指令:\n{prompt}\n\n### 响应:\n"
    }
    
    import json
    config_path = os.path.join(mobile_output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"移动设备模型包已创建: {mobile_output_dir}")
    
    # 创建README.md
    readme = f"""# ISO标准专家移动设备部署包

## 模型信息
- 名称: ISO标准专家
- 基础模型: {args.base_model}
- 量化类型: {args.quantize}
- 上下文大小: 4096 tokens

## 使用方法

### 安装移动应用
在手机上安装以下应用之一：
- iOS: 使用MLC Chat或LLM Lab
- Android: 使用MLC Chat或llama.cpp Android应用

### 导入模型
1. 将 `{model_filename}` 文件和 `config.json` 文件传输到您的移动设备
2. 在应用中导入模型

### 使用提示格式
使用以下格式提问：
```
问题或指令
```

模型会根据ISO标准化文件编写导则回答您的问题。
"""
    
    readme_path = os.path.join(mobile_output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    
    return mobile_output_dir

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 安装必要的依赖
    install_llama_cpp_python()
    
    # 合并LoRA权重（如果需要）
    if args.merge_lora:
        model_path = merge_lora_to_base_model(args)
    else:
        model_path = args.peft_model
    
    # 转换为GGUF格式
    gguf_model_path = convert_to_gguf(model_path, args)
    
    # 创建移动设备可用的模型包
    mobile_output_dir = create_mobile_package(gguf_model_path, args)
    
    print(f"""
处理完成!
- 导出的模型: {gguf_model_path}
- 移动设备包: {mobile_output_dir}

您可以使用以下工具在手机上使用此模型:
- iOS: MLC Chat, LLM Lab
- Android: MLC Chat, llama.cpp Android

请参阅 {os.path.join(mobile_output_dir, "README.md")} 获取更多信息。
    """)

if __name__ == "__main__":
    main() 