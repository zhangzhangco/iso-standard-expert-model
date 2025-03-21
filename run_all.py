#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import time
import json
import glob

def load_config(config_file="config.json"):
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def run_command(command, description=None):
    """运行命令并打印输出"""
    if description:
        print(f"\n=== {description} ===")
    
    print(f"执行: {command}")
    start_time = time.time()
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"命令执行失败，返回码: {process.returncode}")
        return False
    else:
        print(f"命令执行成功，耗时: {end_time - start_time:.2f} 秒")
        return True

def parse_args():
    parser = argparse.ArgumentParser(description="ISO标准专家模型训练和导出流程")
    parser.add_argument("--html_file", type=str, default="data/iso_directives.html", help="ISO标准HTML文件")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-r1-distill-qwen-7b", help="基础模型名称")
    parser.add_argument("--max_sections", type=int, default=100, help="生成问答时处理的最大章节数")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA的rank值")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA的alpha值")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--use_4bit", action="store_true", help="是否使用4位量化训练")
    parser.add_argument("--bf16", action="store_true", help="是否使用bf16精度")
    parser.add_argument("--output_dir", type=str, default="./models", help="输出目录")
    parser.add_argument("--skip_data_prep", action="store_true", help="跳过数据准备步骤")
    parser.add_argument("--skip_training", action="store_true", help="跳过训练步骤")
    parser.add_argument("--skip_evaluation", action="store_true", help="跳过评估步骤")
    parser.add_argument("--skip_export", action="store_true", help="跳过导出步骤")
    parser.add_argument("--skip_model_download", action="store_true", help="跳过模型下载步骤")
    parser.add_argument("--skip_config_fix", action="store_true", help="跳过配置修复步骤")
    parser.add_argument("--config_file", type=str, default="config.json", help="配置文件路径")
    
    return parser.parse_args()

def clean_temp_files():
    """清理可能生成的临时文件"""
    # 查找并删除以=开头的临时文件
    temp_files = glob.glob("=*")
    if temp_files:
        print(f"\n清理临时文件: {', '.join(temp_files)}")
        for file in temp_files:
            try:
                os.remove(file)
                print(f"已删除: {file}")
            except Exception as e:
                print(f"无法删除 {file}: {e}")
    else:
        print("\n没有发现需要清理的临时文件")
    
    # 确保debug_files目录存在
    debug_dir = "debug_files"
    if os.path.exists(debug_dir) and os.path.isdir(debug_dir):
        debug_files = glob.glob(os.path.join(debug_dir, "debug_*.txt"))
        if debug_files:
            print(f"\n清理debug文件: 共{len(debug_files)}个文件")
            for file in debug_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"无法删除 {file}: {e}")
            print(f"已清理debug_files目录中的所有debug文件")
        else:
            print("\ndebug_files目录中没有发现需要清理的文件")

def main():
    args = parse_args()
    
    # 加载配置文件
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
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 准备数据
    if not args.skip_data_prep:
        # 检查是否已存在问答数据集
        if os.path.exists("data/iso_alpaca_dataset.json"):
            print("发现已存在的问答数据集 data/iso_alpaca_dataset.json，将直接使用")
        else:
            # 下载HTML文件（如果不存在）
            if not args.skip_model_download and not os.path.exists(args.html_file):
                run_command(
                    f"wget -O {args.html_file} https://www.iso.org/sites/directives/current/part2/index.xhtml",
                    description="下载ISO标准HTML文件"
                )
            
            # 提取章节
            run_command(
                "python scripts/extract_iso_content.py",
                description="从HTML提取章节"
            )
            
            # 生成问答数据集
            run_command(
                f"python scripts/generate_qa_dataset.py --sections_file data/iso_sections.json --output_file data/iso_alpaca_dataset.json --max_sections {args.max_sections} --config {args.config_file}",
                description="生成问答数据集"
            )
    else:
        print("跳过数据准备步骤")
    
    # 2. 设置环境和下载模型
    if not args.skip_training:
        setup_command = f"python scripts/setup_training.py --model {args.model_name} --config_file {args.config_file}"
        
        if args.skip_model_download:
            setup_command += " --skip_download"
            
        run_command(
            setup_command,
            description="设置训练环境"
        )
        
        # 3. 开始训练
        train_command = (
            f"python scripts/finetune_lora.py "
            f"--model_name {args.model_name} "
            f"--output_dir {args.output_dir} "
            f"--num_epochs {args.num_epochs} "
            f"--learning_rate {args.learning_rate} "
            f"--lora_r {args.lora_r} "
            f"--lora_alpha {args.lora_alpha} "
        )
        
        if args.use_4bit:
            train_command += " --use_4bit"
        
        if args.bf16:
            train_command += " --bf16"
            
        run_command(
            train_command,
            description="LoRA微调模型"
        )
    else:
        print("跳过训练步骤")
    
    # 新增步骤: 创建兼容配置
    if not args.skip_config_fix:
        fixed_dir = f"{args.output_dir}/fixed"
        config_command = f"python scripts/fix_config.py --input_dir {args.output_dir} --output_dir {fixed_dir}"
        run_command(
            config_command,
            description="创建兼容配置"
        )
    else:
        print("跳过配置修复步骤")
    
    # 4. 评估模型
    if not args.skip_evaluation:
        # 使用兼容配置目录进行评估
        eval_dir = f"{args.output_dir}/fixed" if not args.skip_config_fix else args.output_dir
        eval_command = (
            f"python scripts/evaluate_model.py "
            f"--base_model {args.model_name} "
            f"--peft_model {eval_dir} "
            f"--output_file {os.path.join(args.output_dir, 'evaluation_results.json')} "
        )
        
        if args.use_4bit:
            eval_command += " --use_4bit"
            
        run_command(
            eval_command,
            description="评估微调后的模型"
        )
    else:
        print("跳过评估步骤")
    
    # 5. 导出模型
    if not args.skip_export:
        # 使用兼容配置目录进行导出
        export_dir = f"{args.output_dir}/fixed" if not args.skip_config_fix else args.output_dir
        export_command = (
            f"python scripts/export_model.py "
            f"--base_model {args.model_name} "
            f"--peft_model {export_dir} "
            f"--output_dir {os.path.join(args.output_dir, 'exported')} "
            f"--merge_lora "
        )
        
        run_command(
            export_command,
            description="导出移动设备可用模型"
        )
    else:
        print("跳过导出步骤")
    
    # 6. 测试模型
    test_dir = f"{args.output_dir}/fixed" if not args.skip_config_fix else args.output_dir
    test_command = f"python iso_expert.py --query \"ISO标准中'shall'和'should'有什么区别？\" --peft_model {test_dir} --temperature 0.3"
    run_command(
        test_command,
        description="测试模型"
    )
    
    print(f"""
=== 处理完成! ===

您可以使用以下方式分享数据集和模型:

1. 数据集:
   - 数据集文件: data/iso_alpaca_dataset.json
   - 上传到Hugging Face: `huggingface-cli upload <your-username>/iso-expert-dataset data/iso_alpaca_dataset.json`

2. 微调后的模型:
   - 模型目录: {test_dir}
   - 上传到Hugging Face: `huggingface-cli upload <your-username>/iso-expert-model {test_dir}`

3. 移动设备模型:
   - 模型目录: {os.path.join(args.output_dir, 'exported/mobile')}
   - 分享此目录中的文件，用户可以在MLC Chat等移动应用中使用

4. 使用模型:
   - 交互模式: `python iso_expert.py --interactive --peft_model {test_dir}`
   - 单次查询: `python iso_expert.py --query "您的问题" --peft_model {test_dir} --temperature 0.3`

祝您使用愉快!
    """)
    
    # 清理临时文件
    clean_temp_files()

if __name__ == "__main__":
    main() 