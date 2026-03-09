source uv_gpt2/bin/activate
DATA_DIR=/mnt/yixiali/CODES/WTY/GPT-2_pretrain/data

python prepare_data.py \
    --source huggingface \
    --output_dir ${DATA_DIR} \
    --num_workers 64