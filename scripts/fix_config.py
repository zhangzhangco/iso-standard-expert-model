#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import shutil

def create_compatible_config(input_dir, output_dir):
    """创建兼容的PEFT配置文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始配置文件
    config_path = os.path.join(input_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 移除可能导致兼容性问题的字段
        problematic_fields = [
            "exclude_modules", 
            "eva_config", 
            "layer_replication", 
            "layers_pattern", 
            "layers_to_transform",
            "megatron_config", 
            "megatron_core", 
            "auto_mapping", 
            "alpha_pattern", 
            "rank_pattern",
            "loftq_config",
            "use_dora",
            "use_rslora"
        ]
        
        # 创建简化的配置
        simplified_config = {}
        for key, value in config.items():
            if key not in problematic_fields:
                simplified_config[key] = value
        
        # 保存简化的配置
        simplified_config_path = os.path.join(output_dir, "adapter_config.json")
        with open(simplified_config_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_config, f, ensure_ascii=False, indent=2)
        
        print(f"已创建兼容的配置文件: {simplified_config_path}")
        
        # 复制模型权重文件
        weights_path = os.path.join(input_dir, "adapter_model.safetensors")
        if os.path.exists(weights_path):
            shutil.copy2(weights_path, os.path.join(output_dir, "adapter_model.safetensors"))
            print(f"已复制模型权重文件到: {output_dir}")
        else:
            print(f"警告: 找不到模型权重文件 {weights_path}")
            return False
        
        # 复制其他必要的文件
        for file in ["training_args.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
            src_path = os.path.join(input_dir, file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(output_dir, file))
                print(f"已复制文件: {file}")
        
        # 创建simplified_adapter_config.json文件
        simplified_adapter_config_path = os.path.join(output_dir, "simplified_adapter_config.json")
        with open(simplified_adapter_config_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_config, f, ensure_ascii=False, indent=2)
        print(f"已创建简化配置文件: {simplified_adapter_config_path}")
        
        return True
    
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="创建兼容的PEFT配置文件")
    parser.add_argument("--input_dir", type=str, default="models", help="输入目录（包含原始adapter_config.json）")
    parser.add_argument("--output_dir", type=str, default="models/fixed", help="输出目录")
    args = parser.parse_args()
    
    success = create_compatible_config(args.input_dir, args.output_dir)
    
    if success:
        print("处理完成! 兼容的模型文件已准备就绪。")
        print(f"您可以使用以下命令查询模型:")
        print(f"python iso_expert.py --interactive --peft_model {args.output_dir}")
    else:
        print("处理失败! 请检查错误信息。")

if __name__ == "__main__":
    main() 