#!/bin/bash
# GPT-2 预训练完整流程脚本

set -e

echo "=========================================="
echo "GPT-2 Pretraining Pipeline"
echo "=========================================="

# 配置参数
MODEL_SIZE=${MODEL_SIZE:-"gpt2-medium"}
DROPOUT=${DROPOUT:-0.1}
DATA_DIR=${DATA_DIR:-"./data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/gpt2-dropout-${DROPOUT}"}

BATCH_SIZE=${BATCH_SIZE:-4}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
MAX_STEPS=${MAX_STEPS:--1}
NUM_EPOCHS=${NUM_EPOCHS:-1}

# Step 1: 安装依赖
echo ""
echo "Step 1: Installing dependencies..."
echo "=========================================="
pip install -r requirements.txt

# Step 2: 准备数据
echo ""
echo "Step 2: Preparing data..."
echo "=========================================="

if [ ! -f "${DATA_DIR}/train.bin" ]; then
    echo "Data not found. Downloading and processing OpenWebText..."
    python prepare_data.py \
        --source huggingface \
        --output_dir ${DATA_DIR} \
        --tokenizer gpt2 \
        --val_ratio 0.01 \
        --num_workers 8
else
    echo "Data already exists at ${DATA_DIR}"
fi

# Step 3: 训练模型
echo ""
echo "Step 3: Training model..."
echo "=========================================="
echo "Model: ${MODEL_SIZE}"
echo "Dropout: ${DROPOUT}"
echo "Batch size: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Learning rate: ${LEARNING_RATE}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# 使用Hugging Face Trainer版本
python train.py \
    --model_size ${MODEL_SIZE} \
    --dropout ${DROPOUT} \
    --from_scratch \
    --train_file ${DATA_DIR}/train.bin \
    --val_file ${DATA_DIR}/val.bin \
    --block_size 1024 \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --fp16

# 或者使用PyTorch原生版本
# python train_native.py \
#     --model_size ${MODEL_SIZE} \
#     --dropout ${DROPOUT} \
#     --from_scratch \
#     --train_file ${DATA_DIR}/train.bin \
#     --val_file ${DATA_DIR}/val.bin \
#     --output_dir ${OUTPUT_DIR} \
#     --num_epochs ${NUM_EPOCHS} \
#     --batch_size ${BATCH_SIZE} \
#     --gradient_accumulation ${GRADIENT_ACCUMULATION} \
#     --learning_rate ${LEARNING_RATE} \
#     --warmup_steps ${WARMUP_STEPS} \
#     --fp16

# Step 4: 评估模型
echo ""
echo "Step 4: Evaluating model..."
echo "=========================================="

# 计算WikiText-2困惑度
echo "Calculating WikiText-2 perplexity..."
python evaluate.py \
    --model_path ${OUTPUT_DIR}/final \
    --eval_type ppl \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1

# 运行lm-evaluation-harness
echo ""
echo "Running lm-evaluation-harness..."
python evaluate.py \
    --model_path ${OUTPUT_DIR}/final \
    --eval_type lm_eval \
    --tasks lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge \
    --batch_size 4

echo ""
echo "=========================================="
echo "Training and evaluation complete!"
echo "Model saved to: ${OUTPUT_DIR}/final"
echo "=========================================="
