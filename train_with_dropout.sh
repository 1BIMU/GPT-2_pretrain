#!/bin/bash
#===============================================================================
# GPT-2 训练脚本 - 带 Dropout (0.1)
#
# 功能：
# 1. 检查数据集是否存在
# 2. 训练带 Dropout=0.1 的 GPT-2
# 3. 支持多卡 DDP 训练
#===============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
export SWANLAB_API_KEY='HLNooXBIzcROXdU2O2SzR'
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

#===============================================================================
# 配置参数（基于nanoGPT和OpenAI论文的最佳实践）
#===============================================================================
MODEL_SIZE=${MODEL_SIZE:-"gpt2-medium"}      # gpt2(124M), gpt2-medium(355M), gpt2-large(774M)
DATA_DIR=${DATA_DIR:-"/outputs/wty/dataset"}
OUTPUT_BASE=${OUTPUT_BASE:-"/outputs/wty/GPT2-output"}

# 训练参数 (nanoGPT验证过的配置)
BATCH_SIZE=${BATCH_SIZE:-12}                 # micro batch size per GPU
LEARNING_RATE=${LEARNING_RATE:-6e-4}         # peak learning rate (nanoGPT标准)
MIN_LR=${MIN_LR:-6e-5}                       # minimum learning rate (peak的1/10)
WARMUP_STEPS=${WARMUP_STEPS:-2000}           # warmup steps
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}            # weight decay (比0.01效果更好)
ADAM_BETA1=${ADAM_BETA1:-0.9}                # Adam beta1
ADAM_BETA2=${ADAM_BETA2:-0.95}               # Adam beta2 (比默认0.999更稳定)
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}          # 梯度裁剪
NUM_EPOCHS=${NUM_EPOCHS:-10}
MAX_STEPS=${MAX_STEPS:--1}                   # -1表示按epoch训练，完整训练建议600000

DROPOUT=0.1                                   # Dropout率

#===============================================================================
# Step 0: 检查环境
#===============================================================================
log_info "Step 0: Checking environment..."

# 检查Python
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please install Python 3.8+ and dependencies (pip install -r requirements.txt)."
    exit 1
fi

# 检查GPU
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')")
    log_success "GPU detected: ${NUM_GPUS}x ${GPU_NAME} (${GPU_MEM})"
else
    NUM_GPUS=0
    log_warn "No GPU detected. Training will be slow on CPU."
fi

# 根据GPU数量自动调整gradient accumulation，保持有效batch size ≈ 480
TARGET_EFFECTIVE_BATCH=480
if [ "${NUM_GPUS}" -gt 0 ]; then
    GRADIENT_ACCUMULATION=$((TARGET_EFFECTIVE_BATCH / (BATCH_SIZE * NUM_GPUS)))
    if [ "${GRADIENT_ACCUMULATION}" -lt 1 ]; then
        GRADIENT_ACCUMULATION=1
    fi
else
    GRADIENT_ACCUMULATION=40
fi
EFFECTIVE_BATCH=$((BATCH_SIZE * GRADIENT_ACCUMULATION * (NUM_GPUS > 0 ? NUM_GPUS : 1)))

# 训练启动命令：多卡用torchrun，单卡用python
if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCH_CMD="torchrun --nproc_per_node=${NUM_GPUS}"
else
    LAUNCH_CMD="python"
fi

#===============================================================================
# 打印配置
#===============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          GPT-2 Training WITH Dropout (0.1)                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
log_info "Configuration:"
echo "  Model:                ${MODEL_SIZE}"
echo "  Dropout:              ${DROPOUT}"
echo "  GPUs:                 ${NUM_GPUS}"
echo "  Launch:               ${LAUNCH_CMD}"
echo "  Batch size:           ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} x ${NUM_GPUS}gpu = ${EFFECTIVE_BATCH}"
echo "  Tokens per batch:     $((EFFECTIVE_BATCH * 1024))"
echo "  Learning rate:        ${LEARNING_RATE} -> ${MIN_LR}"
echo "  Adam betas:           (${ADAM_BETA1}, ${ADAM_BETA2})"
echo "  Warmup steps:         ${WARMUP_STEPS}"
echo "  Weight decay:         ${WEIGHT_DECAY}"
echo "  Grad clip:            ${MAX_GRAD_NORM}"
echo "  Epochs:               ${NUM_EPOCHS}"
echo "  Output:               ${OUTPUT_BASE}/gpt2-dropout-${DROPOUT}"
echo ""

#===============================================================================
# Step 1: 检查数据
#===============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Step 1: Checking data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "${DATA_DIR}/train.bin" ] || [ ! -f "${DATA_DIR}/val.bin" ]; then
    log_error "Data not found at ${DATA_DIR}"
    log_error "Please run prepare_data.py first:"
    log_error "  python prepare_data.py --source huggingface --output_dir ${DATA_DIR}"
    exit 1
fi

log_success "Data found at ${DATA_DIR}"

# 显示数据信息
if [ -f "${DATA_DIR}/meta.json" ]; then
    python -c "
import json
with open('${DATA_DIR}/meta.json') as f:
    meta = json.load(f)
print(f\"  Train tokens: {meta.get('train_tokens', 'N/A'):,}\")
print(f\"  Val tokens:   {meta.get('val_tokens', 'N/A'):,}\")
"
fi

#===============================================================================
# Step 2: 训练模型
#===============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Step 2: Training model WITH Dropout (${DROPOUT})..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OUTPUT_DIR="${OUTPUT_BASE}/gpt2-dropout-${DROPOUT}"

if [ -d "${OUTPUT_DIR}/final" ]; then
    log_warn "Model already exists at ${OUTPUT_DIR}/final"
    read -p "Do you want to resume training or start fresh? (resume/fresh/skip): " choice
    case "$choice" in
        resume)
            RESUME_FLAG="--resume_from_checkpoint auto"
            ;;
        fresh)
            log_warn "Removing existing model..."
            rm -rf "${OUTPUT_DIR}"
            RESUME_FLAG=""
            ;;
        skip)
            log_info "Skipping training."
            exit 0
            ;;
        *)
            log_error "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    RESUME_FLAG=""
fi

${LAUNCH_CMD} train.py \
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
    --weight_decay ${WEIGHT_DECAY} \
    --adam_beta1 ${ADAM_BETA1} \
    --adam_beta2 ${ADAM_BETA2} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --logging_steps 100 \
    --save_steps 5000 \
    --eval_steps 1000 \
    --fp16 \
    ${RESUME_FLAG}

log_success "Training with Dropout=${DROPOUT} complete!"
echo ""
echo "Model saved to: ${OUTPUT_DIR}/final"
echo "To view logs: tensorboard --logdir ${OUTPUT_DIR}/logs"
echo ""
