#!/bin/bash
# GPT-2 对比实验：AGD vs Random Dropout vs No Dropout

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

# 输出目录
OUTPUT_DIR="./output/gpt2"

echo "=========================================="
echo "GPT-2 对比实验"
echo "=========================================="
echo "模型: $MODEL_SIZE"
echo "最大步数: $MAX_STEPS"
echo "实验组:"
echo "  1. AGD"
echo "  2. Random Dropout (p=0.1)"
echo "  3. No Dropout (p=0.0)"
echo "=========================================="

# 1. AGD 模式
echo ""
echo "🔥 [1/3] 启动 AGD 模式训练..."
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
    --gen_steps 3 \
    --drop_cost_base 0.1 \
    --drop_limit 0.1 \
    --limit_penalty 85.0 \
    --entropy_weight 0.1 \
    --task_loss_weight 0.2 \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --max_checkpoints 3

echo ""
echo "✅ AGD 训练完成!"
echo ""

# 2. Random Dropout (p=0.1)
echo "🧊 [2/3] 启动 Random Dropout (p=0.1) 训练..."
accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py \
    --mode random \
    --dropout_p 0.1 \
    --model_size $MODEL_SIZE \
    --from_scratch \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr_model $LR_MODEL \
    --warmup_steps $WARMUP_STEPS \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --max_checkpoints 3

echo ""
echo "✅ Random Dropout (p=0.1) 训练完成!"
echo ""

# 3. No Dropout (p=0.0)
echo "❄️  [3/3] 启动 No Dropout (p=0.0) 训练..."
accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py \
    --mode random \
    --dropout_p 0.0 \
    --model_size $MODEL_SIZE \
    --from_scratch \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr_model $LR_MODEL \
    --warmup_steps $WARMUP_STEPS \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --max_checkpoints 3

echo ""
echo "✅ No Dropout 训练完成!"
echo ""

# 汇总结果
echo "=========================================="
echo "📊 实验完成！结果汇总："
echo "=========================================="
echo ""
echo "AGD 结果:"
cat $OUTPUT_DIR"_agd/results_agd.json"
echo ""
echo "Random Dropout (p=0.1) 结果:"
cat $OUTPUT_DIR"_random_p0.1/results_random_p0.1.json"
echo ""
echo "No Dropout (p=0.0) 结果:"
cat $OUTPUT_DIR"_random_p0.0/results_random_p0.0.json"
echo ""
echo "=========================================="
