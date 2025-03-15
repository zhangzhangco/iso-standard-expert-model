# ISO标准专家模型训练项目

这个项目旨在通过微调DeepSeek 7B模型，创建一个精通ISO标准化文件编写导则的专家模型。该模型可以回答关于ISO标准文档编写的各种问题，并提供符合导则的建议。

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
│   └── export_model.py        # 导出为移动设备可用格式
├── models/                    # 模型文件夹（训练结果）
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
6. 评估模型性能
7. 导出为移动设备可用的模型

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
python scripts/finetune_lora.py --model_name deepseek-ai/deepseek-llm-7b-base --use_4bit --bf16
```

#### 5. 评估模型

```bash
python scripts/evaluate_model.py --base_model deepseek-ai/deepseek-llm-7b-base --peft_model ./models
```

#### 6. 导出模型

```bash
python scripts/export_model.py --base_model deepseek-ai/deepseek-llm-7b-base --peft_model ./models --merge_lora
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
huggingface-cli upload <your-username>/iso-expert-model ./models
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
--num_epochs：训练轮数
--use_4bit：使用4位量化训练
--bf16：使用bf16精度
--output_dir：输出目录
--skip_data_prep：跳过数据准备步骤
--skip_training：跳过训练步骤
--skip_evaluation：跳过评估步骤
--skip_export：跳过导出步骤
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

或修改`run_all.py`中的默认值：

```python
parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-r1-distill-qwen-7b", help="基础模型名称")
```
