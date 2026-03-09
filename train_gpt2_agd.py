"""
GPT-2 训练脚本 (支持 AGD / Random Dropout 对比)

使用方法:
  AGD 模式:
    accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py --mode agd

  Random Dropout:
    accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py --mode random --dropout_p 0.1

  No Dropout:
    accelerate launch --multi_gpu --num_processes=4 train_gpt2_agd.py --mode random --dropout_p 0.0
"""

import torch
import argparse
import os
import shutil
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import math

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

# 导入本地模块
from agd_core import inject_dropout_to_gpt2, AGDDropoutWrapper, compute_gen_loss, _get_grad_per_output
from dataset import PretrainDataset, DataCollatorForLM


def parse_args():
    parser = argparse.ArgumentParser(description='GPT-2 训练 (AGD vs Random Dropout)')
    parser.add_argument('--mode', type=str, default='agd', choices=['agd', 'random'],
                        help='训练模式: agd 或 random')
    parser.add_argument('--dropout_p', type=float, default=0.1,
                        help='Random Dropout 概率 (仅在 mode=random 时生效)')

    # 模型参数
    parser.add_argument('--model_size', type=str, default='gpt2-medium',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--from_scratch', action='store_true', default=True,
                        help='是否从头训练（随机初始化）')

    # 训练参数
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='最大训练步数 (-1 表示按 num_epochs 自动计算)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练 epoch 数 (仅当 max_steps=-1 时生效)')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='每个设备的 batch size')
    parser.add_argument('--grad_accum', type=int, default=40,
                        help='梯度累积步数')
    parser.add_argument('--lr_model', type=float, default=3e-4,
                        help='模型学习率')
    parser.add_argument('--lr_gen', type=float, default=3e-4,
                        help='Generator 学习率 (仅 AGD 模式)')
    parser.add_argument('--min_lr', type=float, default=3e-5,
                        help='最小学习率 (cosine decay 终点)')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='学习率 warmup 步数')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='权重衰减')
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.95)
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='梯度裁剪')

    # 数据参数
    parser.add_argument('--train_file', type=str, default='./data/train.bin')
    parser.add_argument('--val_file', type=str, default='./data/val.bin')
    parser.add_argument('--block_size', type=int, default=1024)

    # 日志和保存
    parser.add_argument('--output_dir', type=str, default='./output/gpt2_agd',
                        help='输出目录基础路径')
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--save_steps', type=int, default=5000)
    parser.add_argument('--max_checkpoints', type=int, default=3)

    # AGD 专用参数
    parser.add_argument('--gen_steps', type=int, default=3,
                        help='每次模型更新对应的 generator 更新次数')
    parser.add_argument('--drop_cost_base', type=float, default=0.1,
                        help='丢弃神经元的基础成本')
    parser.add_argument('--drop_limit', type=float, default=0.1,
                        help='丢弃率的软上限')
    parser.add_argument('--limit_penalty', type=float, default=7.0,
                        help='违反丢弃限制的惩罚系数')
    parser.add_argument('--entropy_weight', type=float, default=0.1,
                        help='熵正则化权重')
    parser.add_argument('--task_loss_weight', type=float, default=0.2,
                        help='任务损失在 generator loss 中的权重')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume 训练的 checkpoint 路径')

    return parser.parse_args()


def build_config(args):
    """根据命令行参数构建配置"""
    config = {
        # --- 核心模式选择 ---
        'mode': args.mode,
        'dropout_p': args.dropout_p,

        # --- 模型参数 ---
        'model_size': args.model_size,
        'from_scratch': args.from_scratch,

        # --- 基础训练参数 ---
        'lr_model': args.lr_model,
        'lr_gen': args.lr_gen,
        'min_lr': args.min_lr,
        'max_steps': args.max_steps,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
        'adam_beta1': args.adam_beta1,
        'adam_beta2': args.adam_beta2,
        'max_grad_norm': args.max_grad_norm,

        # --- 数据参数 ---
        'train_file': args.train_file,
        'val_file': args.val_file,
        'block_size': args.block_size,

        # --- 日志和保存 ---
        'logging_steps': args.logging_steps,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'max_checkpoints': args.max_checkpoints,

        # --- Resume ---
        'resume': args.resume,

        # --- AGD 专用参数 ---
        'gen_steps': args.gen_steps,
        'drop_cost_base': args.drop_cost_base,
        'drop_limit': args.drop_limit,
        'limit_penalty': args.limit_penalty,
        'entropy_weight': args.entropy_weight,
        'task_loss_weight': args.task_loss_weight
    }

    # 根据模式调整输出目录
    if config['mode'] == 'random':
        config['output_dir'] = f"{args.output_dir}_random_p{config['dropout_p']}"
    else:
        config['output_dir'] = f"{args.output_dir}_agd"

    return config


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Cosine 学习率调度器（带 warmup）

    Args:
        optimizer: 优化器
        num_warmup_steps: warmup 步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率 / 初始学习率
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup: 线性增长
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_perplexity(model, val_loader, device, accelerator):
    """在验证集上计算困惑度"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        outputs = model(**batch)
        loss = outputs.loss

        # 累积损失（需要考虑 batch 中的 token 数量）
        batch_size = batch['input_ids'].shape[0]
        seq_len = batch['input_ids'].shape[1]
        total_loss += loss.item() * batch_size * seq_len
        total_tokens += batch_size * seq_len

    # 多卡训练：all_reduce 所有 rank 的损失
    if accelerator.num_processes > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_tokens_tensor = torch.tensor(total_tokens, device=device)
        torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tokens_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss = total_loss_tensor.item()
        total_tokens = total_tokens_tensor.item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    model.train()
    return perplexity, avg_loss


def manage_checkpoints(output_dir, max_keep=3):
    """管理 checkpoint 数量，只保留最新的 max_keep 个"""
    if not os.path.exists(output_dir):
        return
    checkpoints = []
    for d in os.listdir(output_dir):
        full_path = os.path.join(output_dir, d)
        if os.path.isdir(full_path) and "step_" in d:
            checkpoints.append(full_path)
    if len(checkpoints) <= max_keep:
        return
    checkpoints.sort(key=os.path.getmtime)
    to_delete = checkpoints[:-max_keep]
    for path in to_delete:
        try:
            shutil.rmtree(path)
        except:
            pass


def main():
    args = parse_args()
    CONFIG = build_config(args)

    # 增加 NCCL 超时时间
    os.environ.setdefault("NCCL_TIMEOUT", "1800")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=CONFIG['grad_accum'],
    )
    device = accelerator.device

    # === 初始化 SwanLab ===
    if accelerator.is_main_process and HAS_SWANLAB:
        swanlab.init(
            project="GPT2-AGD",
            experiment_name=f"{CONFIG['model_size']}-{CONFIG['mode']}",
            config=CONFIG,
        )
        print("📊 SwanLab 已初始化")

    # 打印当前模式
    if accelerator.is_main_process:
        print(f"🚀 GPT-2 训练 | 模式: {CONFIG['mode'].upper()} | 模型: {CONFIG['model_size']}")
        print(f"📂 输出目录: {CONFIG['output_dir']}")
        print(f"📊 梯度累积: {CONFIG['grad_accum']} 步")
        effective_batch = CONFIG['batch_size'] * CONFIG['grad_accum'] * accelerator.num_processes
        print(f"📊 有效 batch size: {CONFIG['batch_size']} × {CONFIG['grad_accum']} × {accelerator.num_processes} = {effective_batch}")
        print(f"📊 有效 tokens/batch: {effective_batch * CONFIG['block_size']:,}")

    # === 创建模型 ===
    # GPT-2 各尺寸的离线 config（无需联网）
    GPT2_CONFIGS = {
        'gpt2':        dict(n_embd=768,  n_layer=12, n_head=12),  # 124M
        'gpt2-medium': dict(n_embd=1024, n_layer=24, n_head=16),  # 355M
        'gpt2-large':  dict(n_embd=1280, n_layer=36, n_head=20),  # 774M
        'gpt2-xl':     dict(n_embd=1600, n_layer=48, n_head=25),  # 1558M
    }

    try:
        model_config = GPT2Config.from_pretrained(CONFIG['model_size'])
    except (OSError, Exception) as e:
        if accelerator.is_main_process:
            print(f"⚠️ 无法从 HuggingFace 加载 config ({e.__class__.__name__})，使用离线 config")
        if CONFIG['model_size'] not in GPT2_CONFIGS:
            raise ValueError(f"未知模型: {CONFIG['model_size']}，支持: {list(GPT2_CONFIGS.keys())}")
        cfg = GPT2_CONFIGS[CONFIG['model_size']]
        model_config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=cfg['n_embd'],
            n_layer=cfg['n_layer'],
            n_head=cfg['n_head'],
        )

    # 设置 dropout（这些是模型内置的 dropout，与 AGD 独立）
    model_config.resid_pdrop = 0.0  # 关闭内置 dropout，使用 AGD
    model_config.attn_pdrop = 0.0
    model_config.embd_pdrop = 0.0

    if CONFIG['from_scratch']:
        model = GPT2LMHeadModel(model_config)
        if accelerator.is_main_process:
            print("✅ 模型已随机初始化 (未��载预训练权重)")
    else:
        model = GPT2LMHeadModel.from_pretrained(CONFIG['model_size'], config=model_config)
        if accelerator.is_main_process:
            print("✅ 模型已加载预训练权重")

    # 启用梯度检查点（节省显存）
    model.gradient_checkpointing_enable()

    # === 注入 AGD 或 Random Dropout ===
    model, generator = inject_dropout_to_gpt2(model, CONFIG)

    # === 创建数据集 ===
    if accelerator.is_main_process:
        print(f"📊 加载数据集...")

    train_dataset = PretrainDataset(CONFIG['train_file'], CONFIG['block_size'])
    val_dataset = PretrainDataset(CONFIG['val_file'], CONFIG['block_size'])
    data_collator = DataCollatorForLM()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=data_collator,
    )

    # === 自动计算 max_steps (如果传了 -1) ===
    if CONFIG['max_steps'] <= 0:
        # steps_per_epoch = ceil(dataset_samples / (batch_size * num_gpus)) / grad_accum
        samples_per_epoch = len(train_dataset)
        batches_per_epoch = math.ceil(samples_per_epoch / (CONFIG['batch_size'] * accelerator.num_processes))
        steps_per_epoch = math.ceil(batches_per_epoch / CONFIG['grad_accum'])
        CONFIG['max_steps'] = steps_per_epoch * CONFIG['num_epochs']
        if accelerator.is_main_process:
            print(f"📊 自动计算 max_steps: {samples_per_epoch:,} samples × {CONFIG['num_epochs']} epochs")
            print(f"   = {steps_per_epoch:,} steps/epoch × {CONFIG['num_epochs']} epochs = {CONFIG['max_steps']:,} total steps")

    # === 优化器设置 ===
    model_params = list(model.parameters())

    if CONFIG['mode'] == 'agd':
        gen_params = list(generator.parameters())
        generator = generator.to(device)
        opt_gen = AdamW(gen_params, lr=CONFIG['lr_gen'])
        gen_scaler = torch.amp.GradScaler('cuda') if accelerator.mixed_precision == 'fp16' else None
    else:
        gen_params = []
        opt_gen = None
        gen_scaler = None

    if accelerator.is_main_process:
        print(f"📊 模型参数数量: {sum(p.numel() for p in model_params):,}")
        if CONFIG['mode'] == 'agd':
            print(f"📊 Generator 参数数量: {sum(p.numel() for p in gen_params):,}")

    opt_model = AdamW(
        model_params,
        lr=CONFIG['lr_model'],
        betas=(CONFIG['adam_beta1'], CONFIG['adam_beta2']),
        weight_decay=CONFIG['weight_decay']
    )

    # 学习率调度器
    min_lr_ratio = CONFIG['min_lr'] / CONFIG['lr_model']
    scheduler = get_cosine_schedule_with_warmup(
        opt_model,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=CONFIG['max_steps'],
        min_lr_ratio=min_lr_ratio
    )

    # Prepare
    model, opt_model, train_loader, val_loader, scheduler = accelerator.prepare(
        model, opt_model, train_loader, val_loader, scheduler
    )

    # === Resume 逻辑 ===
    start_step = 0
    best_perplexity = float('inf')

    if CONFIG['resume']:
        checkpoint_path = CONFIG['resume']
        if accelerator.is_main_process:
            print(f"📂 Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(checkpoint['model_state_dict'])

        if CONFIG['mode'] == 'agd' and generator is not None and checkpoint.get('generator_state_dict'):
            generator.load_state_dict(checkpoint['generator_state_dict'])

        opt_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
        if CONFIG['mode'] == 'agd' and opt_gen is not None and checkpoint.get('optimizer_gen_state_dict'):
            opt_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_step = checkpoint['step']
        best_perplexity = checkpoint.get('best_perplexity', float('inf'))

        if accelerator.is_main_process:
            print(f"✅ Resumed from step {start_step}, best_perplexity={best_perplexity:.2f}")

        del checkpoint
        accelerator.wait_for_everyone()

    # === 训练循环 ===
    model.train()
    if accelerator.is_main_process:
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        print(f"📊 Max steps: {CONFIG['max_steps']}")

    accelerator.wait_for_everyone()

    global_step = start_step
    loss_ema = None
    ema_decay = 0.9
    last_gen_loss = 0.0  # 最近一次 gen_loss（用于日志）
    last_cost_val = 0.0  # 最近一次 cost（用于日志）

    # 创建无限循环的数据加载器
    from itertools import cycle
    train_iter = cycle(train_loader)

    progress_bar = tqdm(range(start_step, CONFIG['max_steps']), disable=not accelerator.is_local_main_process)

    while global_step < CONFIG['max_steps']:
        batch = next(train_iter)

        # ==========================================
        # Phase A: Attack (仅 AGD 模式)
        # 🚨 修复：只有在梯度真正同步时才更新 Generator，实现公平对决
        # ==========================================
        cost_val = 0.0
        if CONFIG['mode'] == 'agd' and accelerator.sync_gradients:
            # 冻结模型，只训练 Generator
            for p in model_params:
                p.requires_grad = False

            for m in model.modules():
                if isinstance(m, AGDDropoutWrapper):
                    m.phase_a = True

            unwrapped = accelerator.unwrap_model(model)
            for gs in range(CONFIG['gen_steps']):
                opt_gen.zero_grad()

                with accelerator.autocast():
                    outputs = unwrapped(**batch)
                    task_loss = outputs.loss

                # 🚨 NaN 同步屏障：防止部分 rank 有 NaN 而其他没有
                is_nan = torch.isnan(task_loss).any()
                if accelerator.num_processes > 1:
                    nan_tensor = torch.tensor(1.0 if is_nan else 0.0, device=device)
                    torch.distributed.all_reduce(nan_tensor, op=torch.distributed.ReduceOp.MAX)
                    global_has_nan = nan_tensor.item() > 0
                else:
                    global_has_nan = is_nan.item() if isinstance(is_nan, torch.Tensor) else is_nan

                if global_has_nan:
                    for m in unwrapped.modules():
                        if isinstance(m, AGDDropoutWrapper):
                            m.stats = {'mask': None, 'probs': None, 'logits': None}
                    continue

                gen_loss, cost_val = compute_gen_loss(unwrapped, task_loss, CONFIG)

                if gen_scaler is not None:
                    gen_scaler.scale(gen_loss).backward()
                else:
                    gen_loss.backward()

                # 🚨 None 梯度保护：避免 clip_grad_norm 和 all_reduce 报错
                for p in gen_params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)

                # 恢复缩放
                if gen_scaler is not None:
                    gen_scaler.unscale_(opt_gen)

                # 🚨 Inf 梯度隔离网：防止 NaN/Inf 梯度通过 all_reduce 感染所有卡
                local_has_inf = False
                for p in gen_params:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        local_has_inf = True
                        break

                if accelerator.num_processes > 1:
                    inf_tensor = torch.tensor(1.0 if local_has_inf else 0.0, device=device)
                    torch.distributed.all_reduce(inf_tensor, op=torch.distributed.ReduceOp.MAX)
                    global_has_inf = inf_tensor.item() > 0
                else:
                    global_has_inf = local_has_inf

                if global_has_inf:
                    opt_gen.zero_grad()
                    if gen_scaler is not None:
                        gen_scaler.update()
                    for m in unwrapped.modules():
                        if isinstance(m, AGDDropoutWrapper):
                            m.stats = {'mask': None, 'probs': None, 'logits': None}
                    continue

                # 安全的跨卡同步
                if accelerator.num_processes > 1:
                    for p in gen_params:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)

                if gen_scaler is not None:
                    gen_scaler.step(opt_gen)
                    gen_scaler.update()
                else:
                    opt_gen.step()

                opt_gen.zero_grad()

                # 清理 stats
                for m in unwrapped.modules():
                    if isinstance(m, AGDDropoutWrapper):
                        m.stats = {'mask': None, 'probs': None, 'logits': None}

            for m in model.modules():
                if isinstance(m, AGDDropoutWrapper):
                    m.phase_a = False

            # 恢复 model_params 的 requires_grad
            for p in model_params:
                p.requires_grad = True

            # 记录最新的 gen_loss 和 cost（用于日志，避免非 sync 步时变量未定义）
            try:
                last_gen_loss = gen_loss.item() if isinstance(gen_loss, torch.Tensor) else 0.0
                last_cost_val = cost_val
            except NameError:
                pass  # 全部 gen_steps 都因 NaN/Inf 跳过，保留上一次的值

        # ==========================================
        # Phase B: Defend / Standard Train
        # ==========================================
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss_model = outputs.loss

            accelerator.backward(loss_model)

            loss_val = loss_model.item()
            if not torch.isnan(loss_model):
                if loss_ema is None:
                    loss_ema = loss_val
                else:
                    loss_ema = ema_decay * loss_ema + (1 - ema_decay) * loss_val

            # 更新进度条
            if accelerator.is_local_main_process:
                ema_str = f"{loss_ema:.4f}" if loss_ema is not None else "N/A"
                desc = f"Step {global_step} Loss: {loss_val:.4f} (EMA: {ema_str})"
                if CONFIG['mode'] == 'agd':
                    drop_rates = []
                    for module in model.modules():
                        if isinstance(module, AGDDropoutWrapper) and module.stats['probs'] is not None:
                            drop_rates.append((1.0 - module.stats['probs'].mean()).item())
                    avg_drop = sum(drop_rates) / len(drop_rates) if drop_rates else 0.0
                    desc += f" | Cost: {last_cost_val:.4f} | Drop: {avg_drop*100:.1f}%"
                if torch.isnan(loss_model):
                    desc += " | ⚠️ NaN"
                progress_bar.set_description(desc)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model_params, max_norm=CONFIG['max_grad_norm'])
            opt_model.step()
            scheduler.step()

            # 更新 grad_ema
            if CONFIG['mode'] == 'agd' and accelerator.sync_gradients:
                for m in model.modules():
                    if isinstance(m, AGDDropoutWrapper):
                        if m.upstream_linear is not None:
                            grad_signal = _get_grad_per_output(m.upstream_linear)
                            if grad_signal is not None:
                                grad_signal = grad_signal / CONFIG['grad_accum']
                                m.update_grad_ema(grad_signal)

                # 同步所有 rank 的 grad_ema
                if accelerator.num_processes > 1:
                    for m in model.modules():
                        if isinstance(m, AGDDropoutWrapper) and m._grad_ema is not None:
                            torch.distributed.all_reduce(m.grad_ema, op=torch.distributed.ReduceOp.AVG)

            opt_model.zero_grad()

        # === SwanLab 日志 (必须在 stats 清空之前) ===
        if global_step % CONFIG['logging_steps'] == 0 and accelerator.is_main_process and HAS_SWANLAB:
            log_dict = {
                "train/loss": loss_val,
                "train/loss_ema": loss_ema if loss_ema is not None else loss_val,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/step": global_step,
            }

            if CONFIG['mode'] == 'agd':
                log_dict["agd/gen_loss"] = last_gen_loss
                log_dict["agd/cost"] = last_cost_val

                # 收集各层 dropout rate (stats 尚未清空)
                unwrapped_for_log = accelerator.unwrap_model(model)
                layer_drop_rates = {}
                all_rates = []
                for m in unwrapped_for_log.modules():
                    if isinstance(m, AGDDropoutWrapper) and m.stats.get('probs') is not None:
                        dr = (1.0 - m.stats['probs'].mean()).item()
                        layer_drop_rates[f"dropout/{m.name}"] = dr
                        all_rates.append(dr)

                log_dict.update(layer_drop_rates)
                if all_rates:
                    log_dict["dropout/avg_rate"] = sum(all_rates) / len(all_rates)
                    log_dict["dropout/max_rate"] = max(all_rates)
                    log_dict["dropout/min_rate"] = min(all_rates)

            swanlab.log(log_dict, step=global_step)

        # 清理 stats
        if CONFIG['mode'] == 'agd':
            for m in model.modules():
                if isinstance(m, AGDDropoutWrapper):
                    m.stats = {'mask': None, 'probs': None, 'logits': None}

        global_step += 1
        progress_bar.update(1)

        # === 评估 ===
        if global_step % CONFIG['eval_steps'] == 0:
            accelerator.wait_for_everyone()

            # 🚨 所有 rank 都必须参与 evaluate（内部有 all_reduce）
            unwrapped = accelerator.unwrap_model(model)
            perplexity, avg_loss = evaluate_perplexity(unwrapped, val_loader, device, accelerator)

            if accelerator.is_main_process:
                print(f"\n🔍 Eval step {global_step}: Perplexity: {perplexity:.2f} | Loss: {avg_loss:.4f}")

                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    print(f"   ✅ New best perplexity: {best_perplexity:.2f}")

                if HAS_SWANLAB:
                    swanlab.log({
                        "eval/perplexity": perplexity,
                        "eval/loss": avg_loss,
                        "eval/best_perplexity": best_perplexity,
                    }, step=global_step)

            accelerator.wait_for_everyone()

        # === 保存 checkpoint ===
        if global_step % CONFIG['save_steps'] == 0:
            accelerator.wait_for_everyone()

            save_path = os.path.join(CONFIG['output_dir'], f"step_{global_step}")
            if accelerator.is_main_process:
                print(f"  💾 Saving checkpoint to {save_path}...")
                os.makedirs(save_path, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                torch.save({
                    'step': global_step,
                    'model_state_dict': unwrapped.state_dict(),
                    'generator_state_dict': generator.state_dict() if CONFIG['mode'] == 'agd' else None,
                    'optimizer_model_state_dict': opt_model.state_dict(),
                    'optimizer_gen_state_dict': opt_gen.state_dict() if CONFIG['mode'] == 'agd' else None,
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_ema': loss_ema,
                    'best_perplexity': best_perplexity,
                    'config': CONFIG,
                }, os.path.join(save_path, 'checkpoint.pt'))

                manage_checkpoints(CONFIG['output_dir'], max_keep=CONFIG['max_checkpoints'])
                print(f"  ✅ Checkpoint saved!")

            accelerator.wait_for_everyone()

    # === 训练结束 ===
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n" + "=" * 50)
        print("✅ 训练完成!")
        print(f"📊 总步数: {global_step}")
        print(f"📊 最终训练 Loss EMA: {loss_ema:.4f}" if loss_ema is not None else "📊 最终训练 Loss EMA: N/A")
        print(f"📊 最佳验证 Perplexity: {best_perplexity:.2f}")
        print("=" * 50)

    # 最终评估 — 所有 rank 都必须参与 (内部有 all_reduce)
    unwrapped = accelerator.unwrap_model(model)
    final_perplexity, final_loss = evaluate_perplexity(unwrapped, val_loader, device, accelerator)

    if accelerator.is_main_process:
        print(f"\n🔍 Final Perplexity: {final_perplexity:.2f} | Loss: {final_loss:.4f}")

        # 保存结果
        import json
        output_json = {
            "model": f"{CONFIG['model_size']} - {CONFIG['mode'].upper()}",
            "max_steps": CONFIG['max_steps'],
            "final_loss_ema": round(loss_ema, 4) if loss_ema is not None else None,
            "best_perplexity": round(best_perplexity, 2),
            "final_perplexity": round(final_perplexity, 2),
            "batch_size": CONFIG['batch_size'],
            "grad_accum": CONFIG['grad_accum'],
            "effective_batch": CONFIG['batch_size'] * CONFIG['grad_accum'] * accelerator.num_processes,
        }

        if CONFIG['mode'] == 'agd':
            json_filename = "results_agd.json"
        else:
            json_filename = f"results_random_p{CONFIG['dropout_p']}.json"

        json_path = os.path.join(CONFIG['output_dir'], json_filename)
        with open(json_path, 'w') as f:
            json.dump(output_json, f, indent=2)
        print(f"✅ 结果已保存至: {json_path}")

        if HAS_SWANLAB:
            swanlab.finish()
            print("📊 SwanLab 已关闭")


if __name__ == "__main__":
    main()
