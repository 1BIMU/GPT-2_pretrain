#!/bin/bash
#===============================================================================
# GPT-2 AGD 冒烟测试脚本
#
# 使用 ./data/ 下的小数据集 (1,120 train tokens / 280 val tokens)
# 快速验证完整训练流程：Phase A (Generator) + Phase B (Model) + Eval + Checkpoint
#
# 用法:
#   bash test.sh          # 默认测试 AGD 模式
#   bash test.sh random   # 测试 Random Dropout 模式
#   bash test.sh both     # 两种模式都测试
#===============================================================================

set -e

cd "$(dirname "$0")"

MODE=${1:-agd}  # agd / random / both

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✅ OK]${NC} $1"; }
log_error()   { echo -e "${RED}[❌ FAIL]${NC} $1"; }

#===============================================================================
# 测试参数（极小规模，纯验证逻辑）
#===============================================================================
MODEL_SIZE="gpt2"                # 最小模型 (124M)
DATA_DIR="./data"                # 小数据集
BLOCK_SIZE=64                    # 小 block_size（1120 tokens → ~17 samples）
BATCH_SIZE=2                     # 小 batch
GRAD_ACCUM=1                     # 无梯度累积
MAX_STEPS=20                     # 只跑 20 步
WARMUP_STEPS=2                   # 极短 warmup
LEARNING_RATE=6e-4
LR_GEN=3e-4
MIN_LR=6e-5
EVAL_STEPS=10                    # 每 10 步评估一次（共会评估 2 次）
SAVE_STEPS=10                    # 每 10 步保存一次
LOGGING_STEPS=5                  # 每 5 步记录一次

# AGD 参数
GEN_STEPS=2
DROP_COST_BASE=0.1
DROP_LIMIT=0.1
LIMIT_PENALTY=7.0
ENTROPY_WEIGHT=0.1
TASK_LOSS_WEIGHT=0.2

OUTPUT_BASE="./output/test_smoke"

#===============================================================================
# 环境检查
#===============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            GPT-2 AGD 冒烟测试 (Smoke Test)                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 检查数据
if [ ! -f "${DATA_DIR}/train.bin" ] || [ ! -f "${DATA_DIR}/val.bin" ]; then
    log_error "小数据集不存在: ${DATA_DIR}/train.bin 或 ${DATA_DIR}/val.bin"
    exit 1
fi
log_info "数据集: $(wc -c < "${DATA_DIR}/train.bin") bytes train / $(wc -c < "${DATA_DIR}/val.bin") bytes val"

# 检查 GPU
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    log_info "GPU: ${NUM_GPUS}x ${GPU_NAME}"
    if [ "${NUM_GPUS}" -gt 1 ]; then
        LAUNCH_CMD="torchrun --nproc_per_node=${NUM_GPUS}"
    else
        LAUNCH_CMD="python"
    fi
else
    NUM_GPUS=0
    LAUNCH_CMD="python"
    # 强制禁用 MPS（macOS Apple Silicon），避免 CPU/MPS 混用问题
    export CUDA_VISIBLE_DEVICES=""
    export ACCELERATE_TORCH_DEVICE="cpu"
    log_info "无 CUDA GPU，强制 CPU 测试"
fi

#===============================================================================
# 运行函数
#===============================================================================
run_test() {
    local test_mode=$1
    local output_dir="${OUTPUT_BASE}/${test_mode}"

    # 训练脚本会自动追加后缀: output_dir_agd 或 output_dir_random_p0.1
    if [ "${test_mode}" = "agd" ]; then
        local actual_output="${output_dir}_agd"
        local result_file="${actual_output}/results_agd.json"
    else
        local actual_output="${output_dir}_random_p0.1"
        local result_file="${actual_output}/results_random_p0.1.json"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local mode_upper=$(echo "${test_mode}" | tr '[:lower:]' '[:upper:]')
    log_info "测试模式: ${mode_upper}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  模型: ${MODEL_SIZE} | block_size: ${BLOCK_SIZE} | batch: ${BATCH_SIZE}"
    echo "  步数: ${MAX_STEPS} | eval间隔: ${EVAL_STEPS} | warmup: ${WARMUP_STEPS}"
    if [ "${test_mode}" = "agd" ]; then
        echo "  gen_steps: ${GEN_STEPS} | drop_limit: ${DROP_LIMIT} | limit_penalty: ${LIMIT_PENALTY}"
    else
        echo "  dropout_p: 0.1"
    fi
    echo ""

    # 清理旧输出
    rm -rf "${actual_output}"

    # 构建命令
    local CMD="${LAUNCH_CMD} train_gpt2_agd.py \
        --mode ${test_mode} \
        --model_size ${MODEL_SIZE} \
        --from_scratch \
        --train_file ${DATA_DIR}/train.bin \
        --val_file ${DATA_DIR}/val.bin \
        --block_size ${BLOCK_SIZE} \
        --output_dir ${output_dir} \
        --max_steps ${MAX_STEPS} \
        --batch_size ${BATCH_SIZE} \
        --grad_accum ${GRAD_ACCUM} \
        --lr_model ${LEARNING_RATE} \
        --lr_gen ${LR_GEN} \
        --min_lr ${MIN_LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --logging_steps ${LOGGING_STEPS} \
        --eval_steps ${EVAL_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --max_checkpoints 2"

    # AGD 专用参数
    if [ "${test_mode}" = "agd" ]; then
        CMD="${CMD} \
            --gen_steps ${GEN_STEPS} \
            --drop_cost_base ${DROP_COST_BASE} \
            --drop_limit ${DROP_LIMIT} \
            --limit_penalty ${LIMIT_PENALTY} \
            --entropy_weight ${ENTROPY_WEIGHT} \
            --task_loss_weight ${TASK_LOSS_WEIGHT}"
    fi

    # Random 专用参数
    if [ "${test_mode}" = "random" ]; then
        CMD="${CMD} --dropout_p 0.1"
    fi

    # 运行
    local start_time=$(date +%s)

    if eval ${CMD}; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo ""
        log_success "${mode_upper} 模式训练完成! (${elapsed}s)"

        # 检查输出文件
        if [ -f "${result_file}" ]; then
            log_success "结果文件已生成: ${result_file}"
            echo "  内容:"
            cat "${result_file}" | sed 's/^/    /'
        else
            log_error "结果文件未生成: ${result_file}"
            return 1
        fi

        # 检查 checkpoint
        if ls -d "${actual_output}"/step_* &>/dev/null; then
            local num_ckpts=$(ls -d "${actual_output}"/step_* | wc -l | tr -d ' ')
            log_success "Checkpoint: ${num_ckpts} 个"
        fi

        return 0
    else
        echo ""
        log_error "${mode_upper} 模式训练失败!"
        return 1
    fi
}

#===============================================================================
# 主流程
#===============================================================================
PASS=0
FAIL=0
start_all=$(date +%s)

# 先跑单元测试
echo ""
log_info "Step 0: 运行单元测试..."
UNIT_TEST_OUTPUT=$(python test_agd_flow.py 2>&1)
if echo "${UNIT_TEST_OUTPUT}" | grep -q "所有测试通过"; then
    log_success "单元测试全部通过"
else
    echo "${UNIT_TEST_OUTPUT}"
    log_error "单元测试失败，请先修复!"
    exit 1
fi

# 按模式运行
case "${MODE}" in
    agd)
        if run_test "agd"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
        ;;
    random)
        if run_test "random"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
        ;;
    both)
        if run_test "agd"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
        if run_test "random"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
        ;;
    *)
        log_error "未知模式: ${MODE}  (可选: agd / random / both)"
        exit 1
        ;;
esac

#===============================================================================
# 汇总
#===============================================================================
end_all=$(date +%s)
total_time=$((end_all - start_all))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    测试结果汇总                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  ✅ 通过: %-3d                                              ║\n" ${PASS}
printf "║  ❌ 失败: %-3d                                              ║\n" ${FAIL}
printf "║  ⏱️  总耗时: %-4ds                                          ║\n" ${total_time}
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ ${FAIL} -gt 0 ]; then
    log_error "存在失败的测试！请检查上方日志。"
    exit 1
else
    log_success "所有冒烟测试通过！可以放心部署到 GPU 集群。"
    echo ""
    echo "下一步:"
    echo "  正式训练: bash train_with_agd.sh --fresh"
    echo ""
fi
