#!/bin/bash
# 对比实验脚本 - 训练带Dropout和不带Dropout的模型

set -e

echo "=========================================="
echo "Dropout Comparison Experiment"
echo "=========================================="

DATA_DIR=${DATA_DIR:-"./data"}
MODEL_SIZE=${MODEL_SIZE:-"gpt2-medium"}

# 检查数据是否存在
if [ ! -f "${DATA_DIR}/train.bin" ]; then
    echo "Error: Data not found at ${DATA_DIR}/train.bin"
    echo "Please run prepare_data.py first."
    exit 1
fi

# 实验1: 带Dropout (0.1)
echo ""
echo "=========================================="
echo "Experiment 1: Training with Dropout=0.1"
echo "=========================================="

python train.py \
    --model_size ${MODEL_SIZE} \
    --dropout 0.1 \
    --from_scratch \
    --train_file ${DATA_DIR}/train.bin \
    --val_file ${DATA_DIR}/val.bin \
    --output_dir ./output/gpt2-dropout-0.1 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 5e-4 \
    --warmup_steps 5000 \
    --fp16

# 实验2: 不带Dropout (0.0)
echo ""
echo "=========================================="
echo "Experiment 2: Training with Dropout=0.0"
echo "=========================================="

python train.py \
    --model_size ${MODEL_SIZE} \
    --dropout 0.0 \
    --from_scratch \
    --train_file ${DATA_DIR}/train.bin \
    --val_file ${DATA_DIR}/val.bin \
    --output_dir ./output/gpt2-dropout-0.0 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --learning_rate 5e-4 \
    --warmup_steps 5000 \
    --fp16

# 评估两个模型
echo ""
echo "=========================================="
echo "Evaluating both models..."
echo "=========================================="

echo ""
echo "Model with Dropout=0.1:"
python evaluate.py \
    --model_path ./output/gpt2-dropout-0.1/final \
    --eval_type both \
    --tasks lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge

echo ""
echo "Model with Dropout=0.0:"
python evaluate.py \
    --model_path ./output/gpt2-dropout-0.0/final \
    --eval_type both \
    --tasks lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge

# 评估原始GPT-2作为基准
echo ""
echo "Original GPT-2 (pretrained by OpenAI):"
python evaluate.py \
    --model_path ${MODEL_SIZE} \
    --eval_type both \
    --tasks lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "=========================================="
