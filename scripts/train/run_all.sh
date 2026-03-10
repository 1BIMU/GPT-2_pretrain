source uv_gpt2/bin/activate

export WANDB_PROJECT=wty_gpt2_pretrain
export DATA_DIR=/mnt/yixiali/CODES/WTY/GPT-2_pretrain/data
export OUTPUT_BASE=/mnt/yixiali/CODES/WTY/GPT-2_pretrain/output

# bash train_with_dropout.sh --resume
# bash train_without_dropout.sh --resume
bash train_with_agd.sh