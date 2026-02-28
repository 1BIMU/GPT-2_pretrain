# GPT-2 预训练实验

从头预训练带Dropout的GPT-2模型，并与OpenAI原始GPT-2进行对比。

## 项目结构

```
GPT-2/
├── config.py           # 配置文件
├── prepare_data.py     # 数据预处理脚本
├── dataset.py          # 数据集类
├── model.py            # 模型定义
├── train.py            # 训练脚本 (Hugging Face Trainer)
├── train_native.py     # 训练脚本 (PyTorch原生)
├── evaluate.py         # 评估脚本
├── run_pretrain.sh     # 完整训练流程
├── run_test.sh         # 快速测试脚本
├── run_comparison.sh   # 对比实验脚本
└── requirements.txt    # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速测试（验证代码）

使用TinyShakespeare小数据集测试：

```bash
chmod +x run_test.sh
./run_test.sh
```

### 3. 完整训练

```bash
# 使用默认参数
chmod +x run_pretrain.sh
./run_pretrain.sh

# 或��定义参数
MODEL_SIZE=gpt2-medium DROPOUT=0.1 BATCH_SIZE=4 ./run_pretrain.sh
```

### 4. 单独运行各步骤

```bash
# 准备数据（从Hugging Face下载OpenWebText）
python prepare_data.py --source huggingface --output_dir ./data

# 训练模型
python train.py \
    --model_size gpt2-medium \
    --dropout 0.1 \
    --train_file ./data/train.bin \
    --val_file ./data/val.bin \
    --output_dir ./output/gpt2-dropout

# 评估模型
python evaluate.py \
    --model_path ./output/gpt2-dropout/final \
    --eval_type both
```

## 对比实验

运行带Dropout和不带Dropout的对比实验：

```bash
chmod +x run_comparison.sh
./run_comparison.sh
```

## 参数说明

### 模型参数
- `--model_size`: gpt2 (124M), gpt2-medium (355M), gpt2-large (774M), gpt2-xl (1.5B)
- `--dropout`: Dropout比例，OpenAI原版为0.0，实验设为0.1

### 训练参数
- `--batch_size`: 每个设备的batch size
- `--gradient_accumulation`: 梯度累积步数
- `--learning_rate`: 学习率
- `--warmup_steps`: 预热步数
- `--fp16`: 使用混合精度训练

## 硬件需求

| 模型 | 显存需求 (FP16) | 建议GPU |
|------|----------------|---------|
| gpt2 (124M) | ~8GB | RTX 3080 |
| gpt2-medium (355M) | ~16GB | RTX 3090/A100 |
| gpt2-large (774M) | ~32GB | A100 40GB |
| gpt2-xl (1.5B) | ~48GB | A100 80GB |

## 评估指标

- WikiText-2 困惑度 (PPL)
- LAMBADA (零样本准确率)
- HellaSwag (零样本准确率)
- PIQA (零样本准确率)
- WinoGrande (零样本准确率)
- ARC-Easy/Challenge (零样本准确率)
