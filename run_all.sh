#!/bin/bash
#===============================================================================
# GPT-2 Dropout对比实验 - 一键启动脚本
#
# 功能：
# 1. 自动下载并处理OpenWebText数据集（划分train/val/test）
# 2. 训练带Dropout(0.1)的GPT-2
# 3. 训练不带Dropout(0.0)的GPT-2
# 4. 在test集上计算PPL
# 5. 运行零样本下游任务评估
# 6. 生成对比报告
#===============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

#===============================================================================
# 配置参数（基于nanoGPT和OpenAI论文的最佳实践）
#===============================================================================
MODEL_SIZE=${MODEL_SIZE:-"gpt2-medium"}      # gpt2(124M), gpt2-medium(355M), gpt2-large(774M)
DATA_DIR=${DATA_DIR:-"./data"}
OUTPUT_BASE=${OUTPUT_BASE:-"./output"}

# 训练参数 (nanoGPT验证过的配置)
BATCH_SIZE=${BATCH_SIZE:-12}                 # micro batch size per GPU
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-40}  # 有效batch = 12*40 = 480
LEARNING_RATE=${LEARNING_RATE:-3e-4}         # peak learning rate
MIN_LR=${MIN_LR:-3e-5}                       # minimum learning rate
WARMUP_STEPS=${WARMUP_STEPS:-2000}           # warmup steps
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}            # weight decay (比0.01效果更好)
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}                   # -1表示按epoch训练，完整训练建议600000

# 数据划分
VAL_RATIO=${VAL_RATIO:-0.005}                # 0.5% 验证集
TEST_RATIO=${TEST_RATIO:-0.005}              # 0.5% 测试集

# 评估任务
EVAL_TASKS=${EVAL_TASKS:-"lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge"}

# 结果文件
RESULTS_FILE="${OUTPUT_BASE}/comparison_results.txt"

#===============================================================================
# 打印配置
#===============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          GPT-2 Dropout Comparison Experiment                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
log_info "Configuration:"
echo "  Model:                ${MODEL_SIZE}"
echo "  Batch size:           ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "  Tokens per batch:     $((BATCH_SIZE * GRADIENT_ACCUMULATION * 1024))"
echo "  Learning rate:        ${LEARNING_RATE} -> ${MIN_LR}"
echo "  Warmup steps:         ${WARMUP_STEPS}"
echo "  Weight decay:         ${WEIGHT_DECAY}"
echo "  Epochs:               ${NUM_EPOCHS}"
echo "  Data split:           train / val(${VAL_RATIO}) / test(${TEST_RATIO})"
echo "  Output:               ${OUTPUT_BASE}"
echo ""

#===============================================================================
# Step 0: 检查环境
#===============================================================================
log_info "Step 0: Checking environment..."

# 检查Python
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please install Python 3.8+."
    exit 1
fi

# 检查GPU
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')")
    log_success "GPU detected: ${GPU_NAME} (${GPU_MEM})"
else
    log_warn "No GPU detected. Training will be slow on CPU."
fi

# 安装依赖
log_info "Installing dependencies..."
pip install -q -r requirements.txt

#===============================================================================
# Step 1: 准备数据
#===============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Step 1: Preparing data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "${DATA_DIR}/train.bin" ] && [ -f "${DATA_DIR}/val.bin" ] && [ -f "${DATA_DIR}/test.bin" ]; then
    log_success "Data already exists at ${DATA_DIR}"

    # 显示数据信息
    if [ -f "${DATA_DIR}/meta.json" ]; then
        python -c "
import json
with open('${DATA_DIR}/meta.json') as f:
    meta = json.load(f)
print(f\"  Train tokens: {meta.get('train_tokens', 'N/A'):,}\")
print(f\"  Val tokens:   {meta.get('val_tokens', 'N/A'):,}\")
print(f\"  Test tokens:  {meta.get('test_tokens', 'N/A'):,}\")
"
    fi
else
    log_info "Downloading and processing OpenWebText..."
    python prepare_data.py \
        --source huggingface \
        --output_dir ${DATA_DIR} \
        --tokenizer gpt2 \
        --val_ratio ${VAL_RATIO} \
        --test_ratio ${TEST_RATIO} \
        --num_workers 8

    log_success "Data preparation complete!"
fi

#===============================================================================
# Step 2: 训练带Dropout的模型
#===============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Step 2: Training model WITH Dropout (0.1)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OUTPUT_DROPOUT="${OUTPUT_BASE}/gpt2-dropout-0.1"

if [ -d "${OUTPUT_DROPOUT}/final" ]; then
    log_warn "Model already exists at ${OUTPUT_DROPOUT}/final, skipping training."
else
    python train.py \
        --model_size ${MODEL_SIZE} \
        --dropout 0.1 \
        --from_scratch \
        --train_file ${DATA_DIR}/train.bin \
        --val_file ${DATA_DIR}/val.bin \
        --block_size 1024 \
        --output_dir ${OUTPUT_DROPOUT} \
        --num_epochs ${NUM_EPOCHS} \
        --max_steps ${MAX_STEPS} \
        --batch_size ${BATCH_SIZE} \
        --gradient_accumulation ${GRADIENT_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --warmup_steps ${WARMUP_STEPS} \
        --logging_steps 100 \
        --save_steps 5000 \
        --eval_steps 1000 \
        --fp16

    log_success "Training with Dropout complete!"
fi

#===============================================================================
# Step 3: 训练不带Dropout的模型
#===============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Step 3: Training model WITHOUT Dropout (0.0)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OUTPUT_NO_DROPOUT="${OUTPUT_BASE}/gpt2-dropout-0.0"

if [ -d "${OUTPUT_NO_DROPOUT}/final" ]; then
    log_warn "Model already exists at ${OUTPUT_NO_DROPOUT}/final, skipping training."
else
    python train.py \
        --model_size ${MODEL_SIZE} \
        --dropout 0.0 \
        --from_scratch \
        --train_file ${DATA_DIR}/train.bin \
        --val_file ${DATA_DIR}/val.bin \
        --block_size 1024 \
        --output_dir ${OUTPUT_NO_DROPOUT} \
        --num_epochs ${NUM_EPOCHS} \
        --max_steps ${MAX_STEPS} \
        --batch_size ${BATCH_SIZE} \
        --gradient_accumulation ${GRADIENT_ACCUMULATION} \
        --learning_rate ${LEARNING_RATE} \
        --warmup_steps ${WARMUP_STEPS} \
        --logging_steps 100 \
        --save_steps 5000 \
        --eval_steps 1000 \
        --fp16

    log_success "Training without Dropout complete!"
fi

#===============================================================================
# Step 4: 评估所有模型
#===============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Step 4: Evaluating all models..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 创建结果文件
mkdir -p ${OUTPUT_BASE}
echo "╔══════════════════════════════════════════════════════════════════╗" > ${RESULTS_FILE}
echo "║          GPT-2 Dropout Comparison Results                        ║" >> ${RESULTS_FILE}
echo "║          Generated: $(date '+%Y-%m-%d %H:%M:%S')                          ║" >> ${RESULTS_FILE}
echo "╚══════════════════════════════════════════════════════════════════╝" >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

# 评估函数
evaluate_model() {
    local model_path=$1
    local model_name=$2

    echo "" >> ${RESULTS_FILE}
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> ${RESULTS_FILE}
    echo "Model: ${model_name}" >> ${RESULTS_FILE}
    echo "Path:  ${model_path}" >> ${RESULTS_FILE}
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> ${RESULTS_FILE}

    log_info "Evaluating ${model_name}..."

    # 1. 计算OpenWebText测试集PPL
    echo "" >> ${RESULTS_FILE}
    echo "[OpenWebText Test Set PPL]" >> ${RESULTS_FILE}
    python evaluate.py \
        --model_path ${model_path} \
        --eval_type ppl \
        --data_file ${DATA_DIR}/test.bin \
        --device cuda 2>&1 | tee -a ${RESULTS_FILE}

    # 2. 计算WikiText-2 PPL
    echo "" >> ${RESULTS_FILE}
    echo "[WikiText-2 PPL]" >> ${RESULTS_FILE}
    python evaluate.py \
        --model_path ${model_path} \
        --eval_type ppl \
        --dataset wikitext \
        --dataset_config wikitext-2-raw-v1 \
        --device cuda 2>&1 | tee -a ${RESULTS_FILE}

    # 3. 零样本下游任务评估
    echo "" >> ${RESULTS_FILE}
    echo "[Zero-shot Downstream Tasks]" >> ${RESULTS_FILE}
    python evaluate.py \
        --model_path ${model_path} \
        --eval_type lm_eval \
        --tasks ${EVAL_TASKS} \
        --num_fewshot 0 \
        --batch_size 4 \
        --device cuda 2>&1 | tee -a ${RESULTS_FILE}
}

# 评估带Dropout的模型
evaluate_model "${OUTPUT_DROPOUT}/final" "GPT-2 with Dropout=0.1 (Ours)"

# 评估不带Dropout的模型
evaluate_model "${OUTPUT_NO_DROPOUT}/final" "GPT-2 with Dropout=0.0 (Ours)"

# 评估原始GPT-2作为基准
echo "" >> ${RESULTS_FILE}
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> ${RESULTS_FILE}
echo "Model: Original GPT-2 (OpenAI pretrained)" >> ${RESULTS_FILE}
echo "Path:  ${MODEL_SIZE}" >> ${RESULTS_FILE}
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> ${RESULTS_FILE}

log_info "Evaluating Original GPT-2 (OpenAI)..."

echo "" >> ${RESULTS_FILE}
echo "[WikiText-2 PPL]" >> ${RESULTS_FILE}
python evaluate.py \
    --model_path ${MODEL_SIZE} \
    --eval_type ppl \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --device cuda 2>&1 | tee -a ${RESULTS_FILE}

echo "" >> ${RESULTS_FILE}
echo "[Zero-shot Downstream Tasks]" >> ${RESULTS_FILE}
python evaluate.py \
    --model_path ${MODEL_SIZE} \
    --eval_type lm_eval \
    --tasks ${EVAL_TASKS} \
    --num_fewshot 0 \
    --batch_size 4 \
    --device cuda 2>&1 | tee -a ${RESULTS_FILE}

#===============================================================================
# Step 5: 生成对比报告
#===============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Step 5: Generating comparison report..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "" >> ${RESULTS_FILE}
echo "╔══════════════════════════════════════════════════════════════════╗" >> ${RESULTS_FILE}
echo "║                      EXPERIMENT COMPLETE                         ║" >> ${RESULTS_FILE}
echo "╚══════════════════════════════════════════════════════════════════╝" >> ${RESULTS_FILE}

log_success "All evaluations complete!"
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                      EXPERIMENT COMPLETE                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "Models saved to:"
echo "  - With Dropout:    ${OUTPUT_DROPOUT}/final"
echo "  - Without Dropout: ${OUTPUT_NO_DROPOUT}/final"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir ${OUTPUT_BASE}"
echo ""
