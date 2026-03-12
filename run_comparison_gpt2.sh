#!/bin/bash
# GPT-2 对比实验：AGD vs Random Dropout vs No Dropout
# 所有实验组共享完全相同的训练超参数（除 dropout 机制外）

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_TIMEOUT=1800

# ====== 共享训练参数（与 train.py / config.py 完全对齐）======
MODEL_SIZE="gpt2-medium"
MAX_STEPS=-1              # -1 = 按 epoch 自动计算 (与基线一致)
NUM_EPOCHS=1              # 与基线相同: 训 1 个 epoch
BATCH_SIZE=12
GRAD_ACCUM=40
LR_MODEL=3e-4
LR_GEN=3e-4
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

# 日志和保存
LOGGING_STEPS=100
EVAL_STEPS=1000
SAVE_STEPS=5000
MAX_CHECKPOINTS=3

# 共享参数字符串
COMMON_ARGS="\
    --model_size $MODEL_SIZE \
    --from_scratch \
    --max_steps $MAX_STEPS \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr_model $LR_MODEL \
    --min_lr $MIN_LR \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --max_grad_norm $MAX_GRAD_NORM \
    --dropout_p $DROPOUT_P \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --block_size $BLOCK_SIZE \
    --output_dir $OUTPUT_DIR \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --max_checkpoints $MAX_CHECKPOINTS"

echo "=========================================="
echo "GPT-2 对比实验"
echo "=========================================="
echo "模型: $MODEL_SIZE"
echo "最大步数: $MAX_STEPS"
echo "混合精度: fp16 (硬编码)"
echo "有效 batch: $BATCH_SIZE × $GRAD_ACCUM × 4 = $((BATCH_SIZE * GRAD_ACCUM * 4))"
echo "实验组:"
echo "  1. AGD (dropout_p=$DROPOUT_P for scale factor)"
echo "  2. Random Dropout (p=$DROPOUT_P)"
echo "  3. No Dropout (p=0.0)"
echo "=========================================="

# 1. AGD 模式
echo ""
echo "🔥 [1/3] 启动 AGD 模式训练..."
accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py \
    --mode agd \
    $COMMON_ARGS \
    --lr_gen $LR_GEN \
    --gen_steps 3 \
    --drop_cost_base 0.1 \
    --drop_limit 0.1 \
    --limit_penalty 85.0 \
    --entropy_weight 0.1 \
    --task_loss_weight 0.2

echo ""
echo "✅ AGD 训练完成!"
echo ""

# 2. Random Dropout (p=0.1)
echo "🧊 [2/3] 启动 Random Dropout (p=$DROPOUT_P) 训练..."
accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py \
    --mode random \
    $COMMON_ARGS

echo ""
echo "✅ Random Dropout (p=$DROPOUT_P) 训练完成!"
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
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --max_checkpoints $MAX_CHECKPOINTS

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
