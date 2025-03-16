# ISO标准专家模型训练项目

本项目通过微调DeepSeek 7B模型，创建一个精通《ISO标准化文件编写导则》的专家模型。该模型可以回答关于ISO标准文档编写的各种问题，提供符合《导则》的建议。

## 项目概述

本项目旨在构建一个专业的ISO标准文档编写辅助工具，通过深度学习技术，将ISO/IEC Directives, Part 2文档中的规则和知识转化为可交互的智能模型。**模型已成功训练完成，可直接使用models/fixed目录下的模型文件。**

## 核心功能

- 根据ISO/IEC Directives, Part 2文档提取内容和规则
- 生成Alpaca格式的问答数据集
- 使用LoRA高效微调DeepSeek 7B模型
- 导出为GGUF格式，支持在移动设备上运行
- 完整的训练评估流程

## 系统架构

### 文件夹结构

```
.
├── config.json                # 配置文件（包含API密钥和模型设置）
├── data/                      # 数据文件夹
│   ├── iso_directives.html    # 原始ISO标准HTML文件
│   ├── iso_sections.json      # 提取的章节内容
│   └── iso_alpaca_dataset.json # 生成的问答数据集
├── scripts/                   # 脚本文件夹
│   ├── extract_iso_content.py # 从HTML提取章节内容
│   ├── generate_qa_dataset.py # 生成问答数据集
│   ├── setup_training.py      # 设置训练环境
│   ├── finetune_lora.py       # LoRA微调模型
│   ├── evaluate_model.py      # 评估模型性能
│   ├── export_model.py        # 导出为移动设备可用格式
│   └── fix_config.py          # 修复PEFT配置兼容性问题
├── models/                    # 模型文件夹（训练结果）
│   ├── fixed/                 # 修复后的模型文件（推荐使用）
│   ├── checkpoint-*/          # 训练过程中的检查点（可安全删除）
│   └── evaluation_results.json # 模型评估结果
├── iso_expert.py              # ISO专家模型查询工具
└── run_all.py                 # 主运行脚本
```

### 配置系统

项目使用`config.json`文件存储配置信息，包括：

```json
{
  "huggingface": {
    "api_key": "您的Hugging Face API密钥"
  },
  "model_settings": {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 64000
  },
  "proxy": {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
  }
}
```

- `huggingface.api_key`：用于访问Hugging Face模型和上传模型的API密钥
- `model_settings`：生成问答数据集和评估模型时使用的参数设置
- `proxy`：网络代理设置，用于在网络受限环境中连接Hugging Face服务
  - 如果您使用代理工具如Clash或V2Ray，通常默认端口为7890
  - 如果不需要代理，可以将值设为空字符串：`"http": "", "https": ""`

## 环境要求

### 依赖安装

在开始之前，请确保安装所有必要的依赖：

```bash
pip install torch>=2.0.0 transformers>=4.40.0 datasets accelerate peft>=0.6.0 
pip install bitsandbytes>=0.41.1 trl>=0.7.4 deepspeed wandb
pip install sentencepiece huggingface_hub beautifulsoup4 tqdm
```

此外，请确保已安装ollama并加载了适当的模型（如qwq:latest）用于生成数据集。

### 硬件要求

- GPU: 至少8GB VRAM（使用4位量化时）
- RAM: 至少16GB
- 存储空间: 至少20GB

## 使用指南

### 完整训练流程

最简单的方法是使用`run_all.py`脚本执行完整的训练流程：

```bash
python run_all.py --use_4bit --bf16
```

这将执行以下步骤：
1. 下载ISO标准HTML文件
2. 提取章节内容
3. 生成问答数据集
4. 设置训练环境
5. 微调模型
6. 创建兼容配置
7. 评估模型性能
8. 导出为移动设备可用的模型
9. 测试模型

### 模块化执行

如果您想单独执行某些步骤，可以使用以下命令：

#### 1. 提取内容

```bash
python scripts/extract_iso_content.py
```

#### 2. 生成数据集

```bash
python scripts/generate_qa_dataset.py
```

#### 3. 设置环境

```bash
python scripts/setup_training.py --model deepseek-ai/deepseek-llm-7b-base
```

#### 4. 微调模型

```bash
python scripts/finetune_lora.py --model_name deepseek-ai/deepseek-r1-distill-qwen-7b --use_4bit --bf16 --lora_r 16 --lora_alpha 32 --learning_rate 5e-5 --num_epochs 5
```

#### 5. 创建兼容配置

```bash
python scripts/fix_config.py --input_dir models --output_dir models/fixed
```

#### 6. 评估模型

```bash
python scripts/evaluate_model.py --base_model deepseek-ai/deepseek-r1-distill-qwen-7b --peft_model ./models/fixed
```

#### 7. 导出模型

```bash
python scripts/export_model.py --base_model deepseek-ai/deepseek-r1-distill-qwen-7b --peft_model ./models/fixed --merge_lora
```

#### 8. 使用模型

```bash
# 交互模式
python iso_expert.py --interactive --peft_model ./models/fixed

# 单次查询
python iso_expert.py --query "ISO标准中'shall'和'should'有什么区别？" --peft_model ./models/fixed --temperature 0.3
```

## 模型部署

### 推荐使用的模型版本

**强烈建议使用models/fixed目录下的模型文件**，而非其他检查点。fixed目录是通过fix_config.py脚本创建的，目的是解决PEFT配置兼容性问题，确保模型在不同环境中的稳定性。

### 清理检查点文件

训练完成后，可以安全地删除models/checkpoint-*目录以节省存储空间，只需保留models/fixed目录即可。这些检查点是训练过程中的中间状态，在模型完成训练后不再需要。

```bash
# 在Linux/macOS系统中清理检查点
rm -rf models/checkpoint-*

# 在Windows系统中清理检查点
rmdir /s /q models\checkpoint-*
```

### 模型评估结果

模型在ISO标准问答方面，能够回答关于ISO/IEC Directives Part 2的专业问题。评估结果显示模型能够：

- 正确解释ISO标准中的专业术语和规则
- 理解并回答关于标准编写格式的问题
- 提供符合ISO规范的建议和解释
- 处理关于标准条款、附录、表格等具体元素的查询

完整的评估结果可在models/evaluation_results.json文件中查看。

### 分享数据集和模型

您可以使用Hugging Face分享数据集和模型：

```bash
huggingface-cli login
huggingface-cli upload <your-username>/iso-expert-dataset data/iso_alpaca_dataset.json
huggingface-cli upload <your-username>/iso-expert-model ./models/fixed
```

### 移动设备部署

导出的模型可以在以下移动应用中使用：

- iOS: MLC Chat, LLM Lab
- Android: MLC Chat, llama.cpp Android

只需将导出目录中的`models/exported/mobile`文件夹内的文件传输到您的移动设备，然后在应用中导入即可。

## 模型选择

本项目支持以下模型：

1. **首选：DeepSeek-R1-Distill-Qwen-7B**
   - 参数量适中：7B规模，经过4-bit量化后内存占用约4-5GB
   - 推理能力强：继承了DeepSeek-R1的强化学习优化，在文档问答中表现出色
   - 开源支持：MIT协议，完全开源
   - 默认配置：项目默认使用此模型

2. **次选：Qwen2.5-7B**
   - 规模合适：7B模型，量化后同样适配移动设备
   - 性能均衡：多语言能力强，适合中文文档问答
   - 微调成熟：阿里提供详细的技术报告和工具支持

3. **轻量备选：Qwen2.5-1.5B**
   - 超小规模：1.5B参数，4-bit量化后内存占用约1-2GB
   - 适合资源有限的场景

要更改使用的模型，可以通过命令行参数指定：

```bash
python run_all.py --model_name "Qwen/Qwen2.5-7B" --use_4bit --bf16
```

## 训练参数优化

### 优化训练参数

以下是经过验证的高效训练参数配置：

- **学习率**：1e-5，提高训练稳定性
- **梯度累积**：4步，模拟更大批量训练
- **批处理大小**：1或2，减少内存占用
- **学习率调度器**：余弦退火(cosine)，平滑学习率变化
- **预热比例**：0.01，帮助模型在训练初期稳定

### 参数对比

| 参数 | 标准值 | 优化值 | 影响 |
|------|--------|--------|------|
| learning_rate | 5e-5 | 1e-5 | 提高训练稳定性，减少过拟合风险 |
| gradient_accumulation_steps | 1 | 4 | 模拟更大批量，提高训练稳定性 |
| batch_size | 4 | 1-2 | 减少内存占用，适应更长序列 |
| warmup_ratio | 0.03 | 0.01 | 更短的预热阶段，更多时间用于有效学习 |
| lr_scheduler | linear | cosine | 平滑学习率变化，提高训练稳定性 |

### 实施方法

在`finetune_lora.py`脚本中实现这些优化：

```python
# 在finetune_lora.py中的TrainingArguments部分
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=1,  # 降低批处理大小
    gradient_accumulation_steps=4,  # 增加梯度累积步骤
    learning_rate=1e-5,  # 降低学习率
    weight_decay=0.01,
    warmup_ratio=0.01,  # 调整预热比例
    lr_scheduler_type="cosine",  # 使用余弦调度器
    # ... 其他参数保持不变
)
```

## 高级配置选项

主要脚本`run_all.py`支持多种命令行参数来自定义训练过程：

```
--html_file：ISO标准HTML文件路径
--model_name：基础模型名称
--max_sections：处理的最大章节数
--num_epochs：训练轮数（默认5）
--lora_r：LoRA的rank值（默认16）
--lora_alpha：LoRA的alpha值（默认32）
--learning_rate：学习率（默认5e-5）
--use_4bit：使用4位量化训练
--bf16：使用bf16精度
--output_dir：输出目录
--skip_data_prep：跳过数据准备步骤
--skip_training：跳过训练步骤
--skip_evaluation：跳过评估步骤
--skip_export：跳过导出步骤
--skip_config_fix：跳过配置修复步骤
--config_file：配置文件路径（默认为config.json）
```

## 最佳实践

### 数据准备

**数据集选择：**
- 针对文档问答任务，准备高质量的问答对（JSON格式）
- 建议数据集规模在1-5万条之间
- 数据量越大效果越好

**数据清洗：**
- 确保数据无重复、无噪声，答案逻辑清晰
- 检查语言一致性，避免语言混合问题
- 对于ISO标准文档，确保术语使用准确，引用规则正确

### 微调技术

**推荐方法：**
- **LoRA**：高效微调方法，适合7B模型，节省显存
- **QLoRA**：在资源受限时使用，支持4-bit量化微调，进一步降低内存需求
- **全参数微调**：如果有充足算力（24GB+GPU），可获得最佳效果

### 超参数设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 | LoRA/QLoRA微调的最佳值 |
| Batch Size | 1 | 受限于显存，配合梯度累积使用 |
| 梯度累积步骤 | 4-8 | 模拟更大的批处理大小 |
| 训练轮数 | 1-3 | 避免过拟合，小数据集建议1-2轮 |
| 温度(Temperature) | 0.6 | 推理时使用，推荐0.5-0.7范围 |
| Top-p | 0.95 | 与官方基准一致 |
| 上下文长度 | 2048-4096 | 根据文档需求调整 |

## 许可证

本项目基于MIT许可证开源。

## 致谢

- ISO/IEC Directives, Part 2文档
- DeepSeek团队提供的基础模型
- Hugging Face团队提供的transformers和PEFT库
- llama.cpp项目提供的量化和移动部署工具
