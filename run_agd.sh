#!/bin/bash
# GPT-2 AGD 训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_TIMEOUT=1800

# 训练参数
MODEL_SIZE="gpt2-medium"
MAX_STEPS=100000
BATCH_SIZE=12
GRAD_ACCUM=40
LR_MODEL=3e-4
LR_GEN=3e-4
WARMUP_STEPS=2000

# 数据路径
TRAIN_FILE="./data/train.bin"
VAL_FILE="./data/val.bin"
BLOCK_SIZE=1024

# AGD 参数
GEN_STEPS=3
DROP_COST_BASE=0.1
DROP_LIMIT=0.1
LIMIT_PENALTY=85.0
ENTROPY_WEIGHT=0.1
TASK_LOSS_WEIGHT=0.2

# 输出目录
OUTPUT_DIR="./output/gpt2"

echo "=========================================="
echo "GPT-2 AGD 训练"
echo "=========================================="
echo "模型: $MODEL_SIZE"
echo "最大步数: $MAX_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "梯度累积: $GRAD_ACCUM"
echo "=========================================="

# AGD 模式训练
echo ""
echo "🔥 启动 AGD 模式训练..."
accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py \
    --mode agd \
    --model_size $MODEL_SIZE \
    --from_scratch \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr_model $LR_MODEL \
    --lr_gen $LR_GEN \
    --warmup_steps $WARMUP_STEPS \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --gen_steps $GEN_STEPS \
    --drop_cost_base $DROP_COST_BASE \
    --drop_limit $DROP_LIMIT \
    --limit_penalty $LIMIT_PENALTY \
    --entropy_weight $ENTROPY_WEIGHT \
    --task_loss_weight $TASK_LOSS_WEIGHT \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --max_checkpoints 3

echo ""
echo "✅ AGD 训练完成!"
