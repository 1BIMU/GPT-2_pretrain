#!/bin/bash
# GPT-2 Random Dropout 训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_TIMEOUT=1800

# 训练参数（与 train.py / config.py 对齐）
MODEL_SIZE="gpt2-medium"
MAX_STEPS=100000
BATCH_SIZE=12
GRAD_ACCUM=40
LR_MODEL=3e-4
MIN_LR=3e-5
WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
MAX_GRAD_NORM=1.0
SEED=42
DROPOUT_P=0.1

# 数据路径
TRAIN_FILE="./data/train.bin"
VAL_FILE="./data/val.bin"
BLOCK_SIZE=1024

# 输出目录
OUTPUT_DIR="./output/gpt2"

echo "=========================================="
echo "GPT-2 Random Dropout 训练"
echo "=========================================="
echo "模型: $MODEL_SIZE"
echo "最大步数: $MAX_STEPS"
echo "混合精度: fp16"
echo "Dropout 概率: $DROPOUT_P"
echo "Batch size: $BATCH_SIZE × $GRAD_ACCUM × 4 = $((BATCH_SIZE * GRAD_ACCUM * 4))"
echo "=========================================="

# Random Dropout 模式训练
echo ""
echo "🧊 启动 Random Dropout 模式训练..."
accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py \
    --mode random \
    --dropout_p $DROPOUT_P \
    --model_size $MODEL_SIZE \
    --from_scratch \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr_model $LR_MODEL \
    --min_lr $MIN_LR \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --max_grad_norm $MAX_GRAD_NORM \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --max_checkpoints 3

echo ""
echo "✅ Random Dropout 训练完成!"
