"""
GPT-2 预训练主脚本
使用 Hugging Face Trainer
"""
import os
import argparse
import torch
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    get_scheduler
)
from transformers.trainer_callback import TrainerCallback

from config import get_config
from model import create_model, get_tokenizer
from dataset import PretrainDataset, DataCollatorForLM


class LoggingCallback(TrainerCallback):
    """自定义日志回调"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            # 打印关键指标
            if "loss" in logs:
                print(f"Step {state.global_step}: loss={logs['loss']:.4f}", end="")
                if "learning_rate" in logs:
                    print(f", lr={logs['learning_rate']:.2e}", end="")
                print()


def compute_metrics(eval_pred):
    """计算评估指标"""
    import numpy as np

    logits, labels = eval_pred
    # 计算困惑度
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 使用交叉熵计算
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    perplexity = torch.exp(loss).item()

    return {"perplexity": perplexity}


def train(config):
    """训练函数"""
    # 设置随机种子
    torch.manual_seed(config.training.seed)

    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)

    # 创建模型
    model, model_config = create_model(
        model_size=config.model.model_size,
        resid_pdrop=config.model.resid_pdrop,
        attn_pdrop=config.model.attn_pdrop,
        embd_pdrop=config.model.embd_pdrop,
        from_scratch=config.model.from_scratch
    )

    # 获取tokenizer
    tokenizer = get_tokenizer(config.model.model_size)

    # 创建数据集
    print("\nLoading datasets...")
    train_dataset = PretrainDataset(
        config.data.train_file,
        config.data.block_size
    )
    val_dataset = PretrainDataset(
        config.data.val_file,
        config.data.block_size
    )

    # 数据整理器
    data_collator = DataCollatorForLM()

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,

        # 训练轮数
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,

        # Batch size
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,

        # 优化器
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=config.training.lr_scheduler_type,

        # 日志
        logging_dir=os.path.join(config.training.output_dir, "logs"),
        logging_steps=config.training.logging_steps,
        logging_first_step=True,

        # 保存
        save_steps=config.training.save_steps,
        save_total_limit=5,

        # 评估
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # 混合精度
        fp16=config.training.fp16,
        bf16=config.training.bf16,

        # 其他
        seed=config.training.seed,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=True,

        # 报告
        report_to=["tensorboard"],

        # 断点续训
        resume_from_checkpoint=config.training.resume_from_checkpoint,
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback()],
    )

    # 开始训练
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    # 保存最终模型
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config.training.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.training.output_dir, "final"))

    # 最终评估
    print("\nFinal evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    print(f"Final perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])).item():.2f}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Pretrain GPT-2 with Dropout")

    # 模型参数
    parser.add_argument("--model_size", type=str, default="gpt2-medium",
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--from_scratch", action="store_true", default=True)

    # 数据参数
    parser.add_argument("--train_file", type=str, default="./data/train.bin")
    parser.add_argument("--val_file", type=str, default="./data/val.bin")
    parser.add_argument("--block_size", type=int, default=1024)

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output/gpt2-dropout")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--gradient_accumulation", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=1000)

    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=False)

    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # 创建配置
    config = get_config(
        model_size=args.model_size,
        dropout=args.dropout,
        from_scratch=args.from_scratch,
    )

    # 更新配置
    config.data.train_file = args.train_file
    config.data.val_file = args.val_file
    config.data.block_size = args.block_size

    config.training.output_dir = args.output_dir
    config.training.num_train_epochs = args.num_epochs
    config.training.max_steps = args.max_steps
    config.training.per_device_train_batch_size = args.batch_size
    config.training.per_device_eval_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation
    config.training.learning_rate = args.learning_rate
    config.training.warmup_steps = args.warmup_steps
    config.training.weight_decay = args.weight_decay
    config.training.adam_beta1 = args.adam_beta1
    config.training.adam_beta2 = args.adam_beta2
    config.training.max_grad_norm = args.max_grad_norm
    config.training.logging_steps = args.logging_steps
    config.training.save_steps = args.save_steps
    config.training.eval_steps = args.eval_steps
    config.training.fp16 = args.fp16
    config.training.bf16 = args.bf16
    config.training.resume_from_checkpoint = args.resume

    # 打印配置
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"Model: {config.model.model_size}")
    print(f"Dropout: {config.model.resid_pdrop}")
    print(f"From scratch: {config.model.from_scratch}")
    print(f"Block size: {config.data.block_size}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Output dir: {config.training.output_dir}")
    print("=" * 50)

    # 开始训练
    train(config)


if __name__ == "__main__":
    main()
