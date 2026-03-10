#!/bin/bash
#===============================================================================
# GPT-2 训练脚本 - 带 AGD (Adversarial Gumbel Dropout)
#
# 功能：
# 1. 检查数据集是否存在
# 2. 训练带 AGD 的 GPT-2
# 3. 支持多卡 DDP 训练
# 4. 支持断点续训: bash train_with_agd.sh --resume
#===============================================================================

set -e

# 解析命令行参数
RESUME_MODE="ask"  # ask/resume/fresh/skip
for arg in "$@"; do
    case $arg in
        --resume)
            RESUME_MODE="resume"
            shift
            ;;
        --fresh)
            RESUME_MODE="fresh"
            shift
            ;;
        --skip)
            RESUME_MODE="skip"
            shift
            ;;
    esac
done

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
OUTPUT_BASE=${OUTPUT_BASE:-"/outputs/wty/GPT2-output/agd"}

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

# AGD 专用参数 (参考 CLIP 的设置)
GEN_STEPS=${GEN_STEPS:-3}                    # Generator 更新次数
DROP_COST_BASE=${DROP_COST_BASE:-0.1}        # 基础丢弃成本
DROP_LIMIT=${DROP_LIMIT:-0.1}                # 丢弃率软上限
LIMIT_PENALTY=${LIMIT_PENALTY:-85.0}         
ENTROPY_WEIGHT=${ENTROPY_WEIGHT:-0.1}        # 熵正则化权重
TASK_LOSS_WEIGHT=${TASK_LOSS_WEIGHT:-0.2}   # 任务损失权重
LR_GEN=${LR_GEN:-3e-4}                       # Generator 学习率

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
echo "║          GPT-2 Training WITH AGD                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
log_info "Configuration:"
echo "  Model:                ${MODEL_SIZE}"
echo "  Mode:                 AGD (Adversarial Gumbel Dropout)"
echo "  GPUs:                 ${NUM_GPUS}"
echo "  Launch:               ${LAUNCH_CMD}"
echo "  Batch size:           ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} x ${NUM_GPUS}gpu = ${EFFECTIVE_BATCH}"
echo "  Tokens per batch:     $((EFFECTIVE_BATCH * 1024))"
echo "  Learning rate:        ${LEARNING_RATE} -> ${MIN_LR}"
echo "  Generator LR:         ${LR_GEN}"
echo "  Adam betas:           (${ADAM_BETA1}, ${ADAM_BETA2})"
echo "  Warmup steps:         ${WARMUP_STEPS}"
echo "  Weight decay:         ${WEIGHT_DECAY}"
echo "  Grad clip:            ${MAX_GRAD_NORM}"
echo "  Epochs:               ${NUM_EPOCHS}"
echo ""
log_info "AGD Parameters:"
echo "  Generator steps:      ${GEN_STEPS}"
echo "  Drop cost base:       ${DROP_COST_BASE}"
echo "  Drop limit:           ${DROP_LIMIT}"
echo "  Limit penalty:        ${LIMIT_PENALTY}"
echo "  Entropy weight:       ${ENTROPY_WEIGHT}"
echo "  Task loss weight:     ${TASK_LOSS_WEIGHT}"
echo "  Output:               ${OUTPUT_BASE}/gpt2-agd"
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
log_info "Step 2: Training model WITH AGD..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OUTPUT_DIR="${OUTPUT_BASE}/gpt2-agd"

if [ -d "${OUTPUT_DIR}/final" ] || ls -d "${OUTPUT_DIR}"/checkpoint-* &>/dev/null || ls -d "${OUTPUT_DIR}"/step_* &>/dev/null; then
    log_warn "Model already exists at ${OUTPUT_DIR}"

    # 如果是交互模式，询问用户
    if [ "${RESUME_MODE}" = "ask" ]; then
        read -p "Do you want to resume training or start fresh? (resume/fresh/skip): " choice
        RESUME_MODE="$choice"
    fi

    case "${RESUME_MODE}" in
        resume)
            log_info "Resuming training from checkpoint..."
            # 查找最新的 checkpoint
            LATEST_CKPT=$(ls -td "${OUTPUT_DIR}"/step_* 2>/dev/null | head -1)
            if [ -n "${LATEST_CKPT}" ]; then
                RESUME_FLAG="--resume ${LATEST_CKPT}/checkpoint.pt"
                log_info "Found checkpoint: ${LATEST_CKPT}"
            else
                log_warn "No checkpoint found, starting fresh..."
                RESUME_FLAG=""
            fi
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
            log_error "Invalid choice: ${RESUME_MODE}. Use --resume, --fresh, or --skip"
            exit 1
            ;;
    esac
else
    RESUME_FLAG=""
fi

${LAUNCH_CMD} train_gpt2_agd.py \
    --mode agd \
    --model_size ${MODEL_SIZE} \
    --from_scratch \
    --train_file ${DATA_DIR}/train.bin \
    --val_file ${DATA_DIR}/val.bin \
    --block_size 1024 \
    --output_dir ${OUTPUT_DIR} \
    --max_steps ${MAX_STEPS} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRADIENT_ACCUMULATION} \
    --lr_model ${LEARNING_RATE} \
    --lr_gen ${LR_GEN} \
    --min_lr ${MIN_LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --adam_beta1 ${ADAM_BETA1} \
    --adam_beta2 ${ADAM_BETA2} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --gen_steps ${GEN_STEPS} \
    --drop_cost_base ${DROP_COST_BASE} \
    --drop_limit ${DROP_LIMIT} \
    --limit_penalty ${LIMIT_PENALTY} \
    --entropy_weight ${ENTROPY_WEIGHT} \
    --task_loss_weight ${TASK_LOSS_WEIGHT} \
    --logging_steps 100 \
    --save_steps 5000 \
    --eval_steps 1000 \
    --max_checkpoints 3 \
    ${RESUME_FLAG}

log_success "Training with AGD complete!"
echo ""
echo "Model saved to: ${OUTPUT_DIR}"
echo "To view results: cat ${OUTPUT_DIR}/results_agd.json"
echo ""
