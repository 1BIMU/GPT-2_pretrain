#!/bin/bash
# 快速测试脚本 - 使用小数据集验证代码

set -e

echo "=========================================="
echo "Quick Test with TinyShakespeare"
echo "=========================================="

# 创建测试数据目录
mkdir -p ./data/test

# 下载TinyShakespeare
echo "Downloading TinyShakespeare..."
if [ ! -f "./data/test/input.txt" ]; then
    curl -o ./data/test/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

# 处理数据（划分train/val/test）
echo "Processing data..."
python -c "
from transformers import GPT2Tokenizer
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

with open('./data/test/input.txt', 'r') as f:
    text = f.read()

tokens = tokenizer.encode(text)
print(f'Total tokens: {len(tokens):,}')

# 划分训练集、验证集和测试集 (90% / 5% / 5%)
n = len(tokens)
train_end = int(n * 0.9)
val_end = int(n * 0.95)

train_tokens = np.array(tokens[:train_end], dtype=np.uint16)
val_tokens = np.array(tokens[train_end:val_end], dtype=np.uint16)
test_tokens = np.array(tokens[val_end:], dtype=np.uint16)

train_tokens.tofile('./data/test/train.bin')
val_tokens.tofile('./data/test/val.bin')
test_tokens.tofile('./data/test/test.bin')

print(f'Train tokens: {len(train_tokens):,}')
print(f'Val tokens: {len(val_tokens):,}')
print(f'Test tokens: {len(test_tokens):,}')
"

# 训练小模型
echo ""
echo "Training small model (gpt2, ~124M params)..."
python train_native.py \
    --model_size gpt2 \
    --dropout 0.1 \
    --from_scratch \
    --train_file ./data/test/train.bin \
    --val_file ./data/test/val.bin \
    --block_size 256 \
    --output_dir ./output/test \
    --num_epochs 1 \
    --max_steps 500 \
    --batch_size 4 \
    --gradient_accumulation 2 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 200 \
    --fp16

echo ""
echo "Test complete! Check ./output/test for results."
