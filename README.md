# ISO标准专家模型训练项目

这个项目通过微调DeepSeek 7B模型，创建一个精通《ISO标准化文件编写导则》的专家模型。该模型可以回答关于ISO标准文档编写的各种问题，提供符合《导则》的建议。

## 功能特点

- 根据ISO/IEC Directives, Part 2文档提取内容和规则
- 生成Alpaca格式的问答数据集
- 使用LoRA高效微调DeepSeek 7B模型
- 导出为GGUF格式，支持在移动设备上运行
- 完整的训练评估流程

## 文件夹结构

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
├── iso_expert.py              # ISO专家模型查询工具
└── run_all.py                 # 主运行脚本
```

## 配置文件

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

## 安装依赖

在开始之前，请确保安装所有必要的依赖：

```bash
pip install torch>=2.0.0 transformers>=4.40.0 datasets accelerate peft>=0.6.0 
pip install bitsandbytes>=0.41.1 trl>=0.7.4 deepspeed wandb
pip install sentencepiece huggingface_hub beautifulsoup4 tqdm
```

此外，请确保已安装ollama并加载了适当的模型（如qwq:latest）用于生成数据集。

## 使用方法

### 完整流程

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

### 单独步骤

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

## 硬件要求

- GPU: 至少8GB VRAM（使用4位量化时）
- RAM: 至少16GB
- 存储空间: 至少20GB

## 分享和部署

### 分享数据集和模型

您可以使用Hugging Face分享数据集和模型：

```bash
huggingface-cli login
huggingface-cli upload <your-username>/iso-expert-dataset data/iso_alpaca_dataset.json
huggingface-cli upload <your-username>/iso-expert-model ./models/fixed
```

### 在移动设备上使用

导出的模型可以在以下移动应用中使用：

- iOS: MLC Chat, LLM Lab
- Android: MLC Chat, llama.cpp Android

只需将导出目录中的`models/exported/mobile`文件夹内的文件传输到您的移动设备，然后在应用中导入即可。

## 自定义选项

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

## 许可证

本项目基于MIT许可证开源。

## 致谢

- ISO/IEC Directives, Part 2文档
- DeepSeek团队提供的基础模型
- Hugging Face团队提供的transformers和PEFT库
- llama.cpp项目提供的量化和移动部署工具

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

## 中期改进计划

为了提高模型在ISO标准领域的专业性和准确性，我们实施了以下中期改进：

### 1. 增强数据生成

我们改进了数据生成脚本`scripts/generate_qa_dataset.py`，主要改进包括：

- 更专业的提示词，引导模型生成更准确的ISO标准相关问答对
- 优先选择包含关键术语（如"shall"、"should"等）的章节
- 要求生成的答案包含具体的规则编号和引用
- 降低温度值(0.5)，获得更确定性的回答
- 增加每个章节生成的问答对数量(5个)

### 2. 改进微调参数

我们改进了微调脚本`scripts/finetune_lora.py`，主要改进包括：

- 增加LoRA的rank值(r=16)和alpha值(32)，提高模型捕捉复杂专业知识的能力
- 使用更小的学习率(5e-5)，帮助模型更稳定地学习
- 增加训练轮数(5轮)，让模型有更多时间学习
- 改进提示模板格式化，确保模型正确理解输入

### 3. 兼容性配置修复

为了解决PEFT库版本兼容性问题，我们创建了配置修复脚本`scripts/fix_config.py`，自动：

- 移除可能导致兼容性问题的配置字段
- 创建简化的配置文件
- 复制必要的模型文件到新目录

### 4. 改进查询接口

我们改进了`iso_expert.py`脚本，提供更专业的上下文和更好的提示模板：

- 添加ISO标准专家的角色定义
- 降低默认温度(0.3)，获得更确定性的回答
- 支持交互式查询和单次查询模式

### 5. 一键训练流程

我们更新了`run_all.py`脚本，整合所有改进：

```bash
# 使用默认参数运行完整流程
python run_all.py --use_4bit --bf16

# 跳过数据准备，只进行训练
python run_all.py --skip_data_prep --use_4bit --bf16

# 自定义参数
python run_all.py --model_name "deepseek-ai/deepseek-r1-distill-qwen-7b" --max_sections 150 --num_epochs 8 --lora_r 32 --lora_alpha 64 --learning_rate 3e-5 --use_4bit --bf16
```

## 训练参数优化

为进一步提高模型质量，可参考来自于LLaMA-Factory项目和Lightblue团队的多语言推理模型训练经验。

### 1. 优化训练参数

以下是从多语言推理训练配置中借鉴的关键参数：

- **学习率降低**：从5e-5降低到1e-5，提高训练稳定性
- **梯度累积**：增加梯度累积步骤(gradient_accumulation_steps)到4，模拟更大批量训练
- **批处理大小**：保持较小值(1或2)，减少内存占用
- **学习率调度器**：使用余弦退火(cosine)调度器，平滑学习率变化
- **预热比例**：设置为0.01，帮助模型在训练初期稳定

### 2. 使用优化参数的训练命令

```bash
# 使用优化参数进行训练
python run_all.py --model_name deepseek-ai/deepseek-r1-distill-qwen-7b \
  --max_sections 200 \
  --num_epochs 5 \
  --learning_rate 1e-5 \
  --use_4bit \
  --bf16 \
  --output_dir ./models_optimized

# 跳过数据准备，仅使用优化参数训练
python run_all.py --skip_data_prep \
  --model_name deepseek-ai/deepseek-r1-distill-qwen-7b \
  --learning_rate 1e-5 \
  --use_4bit \
  --bf16 \
  --output_dir ./models_optimized
```

### 3. 参数对比

| 参数 | 原始值 | 优化值 | 影响 |
|------|--------|--------|------|
| learning_rate | 5e-5 | 1e-5 | 提高训练稳定性，减少过拟合风险 |
| gradient_accumulation_steps | 1 | 4 | 模拟更大批量，提高训练稳定性 |
| batch_size | 4 | 1-2 | 减少内存占用，适应更长序列 |
| warmup_ratio | 0.03 | 0.01 | 更短的预热阶段，更多时间用于有效学习 |
| lr_scheduler | linear | cosine | 平滑学习率变化，提高训练稳定性 |

### 4. 实施方法

要在`finetune_lora.py`脚本中实现这些优化，可以修改以下参数：

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

这些优化参数可能会（但不确定）适合处理复杂的专业领域知识，如ISO标准文档编写规范，有助于模型更好地理解和应用规则。

## DeepSeek-R1-Distill-Qwen-7B微调可能的最佳实践

基于社区经验和最新实践，以下是针对DeepSeek-R1-Distill-Qwen-7B模型微调的建议：

### 1. 数据准备

**数据集选择：**
- 针对文档问答任务，准备高质量的问答对（JSON格式）
- 建议数据集规模在1-5万条之间（社区经验表明7B模型需要>17k样本才能显著提升性能）
- 官方蒸馏版本使用了约80万样本，数据量越大效果越好

**数据清洗：**
- 确保数据无重复、无噪声，答案逻辑清晰
- 检查语言一致性，避免语言混合问题
- 对于ISO标准文档，确保术语使用准确，引用规则正确

### 2. 微调方法

**推荐技术：**
- **LoRA**：高效微调方法，适合7B模型，节省显存
- **QLoRA**：在资源受限时使用，支持4-bit量化微调，进一步降低内存需求
- **全参数微调**：如果有充足算力（24GB+GPU），可获得最佳效果

**工具推荐：**
- **LLaMA-Factory**：支持DeepSeek模型，提供配置文件模板
- **Unsloth**：速度快，显存占用低，适合资源有限环境
- **Hugging Face Transformers**：基础工具，搭配bitsandbytes用于量化

### 3. 超参数设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 | LoRA/QLoRA微调的最佳值，避免破坏预训练权重 |
| Batch Size | 1 | 受限于显存，配合梯度累积使用 |
| 梯度累积步骤 | 4-8 | 模拟更大的批处理大小 |
| 训练轮数 | 1-3 | 避免过拟合，小数据集建议1-2轮 |
| 温度(Temperature) | 0.6 | 推理时使用，官方推荐0.5-0.7范围 |
| Top-p | 0.95 | 与官方基准一致 |
| 上下文长度 | 2048-4096 | 根据文档需求调整，最大支持32,768 token |
| 学习率调度器 | cosine | 平滑学习率变化，提高训练稳定性 |
| 预热比例 | 0.01 | 帮助模型在训练初期稳定 |

### 4. 训练配置

**硬件要求：**
- 单张24GB GPU（如RTX 4090）可完成7B模型的LoRA微调
- 使用DeepSpeed Zero-2或Zero-3优化显存
- 4-bit量化训练可在16GB显存GPU上运行

**量化：**
- 训练后用bitsandbytes或llama.cpp转为4-bit格式
- 量化后内存占用降至4GB，适配移动设备运行

**提示设计：**
- 避免系统提示，所有指令放入用户提示中
- 对于ISO标准问答，可加入专家角色定义和具体指令

### 5. 验证与优化

**验证集：**
- 分割1%-10%数据作为验证集
- 每0.1-0.5步评估一次（eval_steps）

**基准测试：**
- 针对ISO标准文档问答，自建小规模测试集（约100条问题）
- 评估准确率、术语使用正确性和回答流畅性

**常见问题优化：**
- 重复输出：提高repetition_penalty（如1.1）
- 推理不佳：增加数据集规模，或调整提示明确任务目标
- 术语混淆：增加特定术语的训练样本

### 6. 部署到移动设备

**量化模型：**
- 训练后用llama.cpp转换为GGUF格式（如Q4_0）
- 内存占用约4GB，适合中高端移动设备

**部署工具：**
- Ollama：支持本地运行，适配量化模型
- MLC-LLM：专为移动端优化，支持安卓和iOS

**硬件要求：**
- 推荐8GB+RAM手机（如Snapdragon 8 Gen 2或iPhone 15 Pro）
- 低端设备需测试性能，可能需要进一步量化

### 7. 完整训练命令示例

```bash
# 使用最佳实践参数进行训练
python run_all.py \
  --model_name deepseek-ai/deepseek-r1-distill-qwen-7b \
  --max_sections 250 \
  --num_epochs 3 \
  --learning_rate 1e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --use_4bit \
  --bf16 \
  --output_dir ./models_best_practice

# 仅微调已有数据集
python scripts/finetune_lora.py \
  --model_name deepseek-ai/deepseek-r1-distill-qwen-7b \
  --dataset_path data/iso_alpaca_dataset.json \
  --output_dir ./models_best_practice \
  --num_epochs 3 \
  --learning_rate 1e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type cosine \
  --use_4bit \
  --bf16
```
