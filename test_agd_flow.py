"""
AGD 端到端流程验证脚本

在 CPU/单卡上用微型 GPT-2 + 小数据集跑几步，验证：
1. Conv1D 兼容性（out_features / weight.sum 维度）
2. Dropout 位置对齐（D2 attn_resid, D3 mlp, D4 embd）
3. Phase A (Generator 训练) 前向 + 反向
4. Phase B (Model 训练) 前向 + 反向
5. compute_gen_loss 计算
6. grad_ema 更新
7. 完整训练 step 不崩溃
"""
import os
import sys
import torch
import numpy as np
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(__file__))

from transformers import GPT2LMHeadModel, GPT2Config
from agd_core import (
    inject_dropout_to_gpt2, AGDDropoutWrapper, compute_gen_loss,
    _get_out_features, _get_weight_mag_per_output, _get_grad_per_output,
    SharedMaskGenerator, RandomDropoutWrapper,
)
from dataset import PretrainDataset, DataCollatorForLM


def create_tiny_data(path, num_tokens=2048):
    """在内存中创建极小的测试数据"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokens = np.random.randint(0, 50257, size=num_tokens, dtype=np.uint16)
    tokens.tofile(path)
    return path


def test_conv1d_helpers():
    """测试 Conv1D 兼容性辅助函数"""
    print("=" * 50)
    print("🧪 Test 1: Conv1D 兼容性辅助函数")
    print("=" * 50)

    config = GPT2Config(
        n_embd=64, n_head=2, n_layer=2,
        n_inner=128, vocab_size=1000, n_positions=128,
    )
    model = GPT2LMHeadModel(config)

    c_proj = model.transformer.h[0].mlp.c_proj
    print(f"  c_proj 类型: {type(c_proj).__name__}")
    print(f"  c_proj.weight.shape: {c_proj.weight.shape}")

    out_feat = _get_out_features(c_proj)
    print(f"  _get_out_features: {out_feat}")
    assert out_feat == config.n_embd, f"Expected {config.n_embd}, got {out_feat}"
    print("  ✅ _get_out_features 正确!")

    w_mag = _get_weight_mag_per_output(c_proj)
    print(f"  _get_weight_mag_per_output shape: {w_mag.shape}")
    assert w_mag.shape[0] == config.n_embd, f"Expected [{config.n_embd}], got {w_mag.shape}"
    print("  ✅ _get_weight_mag_per_output 维度正确!")

    n_inner = config.n_inner if config.n_inner is not None else 4 * config.n_embd
    x = torch.randn(1, 10, n_inner)
    out = c_proj(x)
    out.sum().backward()
    grad_signal = _get_grad_per_output(c_proj)
    print(f"  _get_grad_per_output shape: {grad_signal.shape}")
    assert grad_signal.shape[0] == config.n_embd, f"Expected [{config.n_embd}], got {grad_signal.shape}"
    print("  ✅ _get_grad_per_output 维度正确!")
    print()


def test_dropout_alignment():
    """测试 dropout 位置与官方 GPT-2 完全对齐"""
    print("=" * 50)
    print("🧪 Test 2: Dropout 位置对齐 (AGD 模式)")
    print("=" * 50)

    config = GPT2Config(
        n_embd=64, n_head=2, n_layer=2,
        n_inner=128, vocab_size=1000, n_positions=128,
        resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)

    # 记录原始 dropout 位置
    print("  原始 GPT-2 dropout 位置:")
    print(f"    D1: attn.attn_dropout   = {type(model.transformer.h[0].attn.attn_dropout).__name__}")
    print(f"    D2: attn.resid_dropout  = {type(model.transformer.h[0].attn.resid_dropout).__name__}")
    print(f"    D3: mlp.dropout         = {type(model.transformer.h[0].mlp.dropout).__name__}")
    print(f"    D4: transformer.drop    = {type(model.transformer.drop).__name__}")

    agd_config = {
        'mode': 'agd', 'dropout_p': 0.1,
        'gen_steps': 2, 'drop_cost_base': 0.1, 'drop_limit': 0.1,
        'limit_penalty': 7.0, 'entropy_weight': 0.1, 'task_loss_weight': 0.2,
    }
    model, generator = inject_dropout_to_gpt2(model, agd_config)

    print("\n  注入后 dropout 位置:")
    # D1: attn_dropout 应保持为原始 Dropout (不注入)
    d1_type = type(model.transformer.h[0].attn.attn_dropout).__name__
    print(f"    D1: attn.attn_dropout   = {d1_type} (保持原始)")
    assert d1_type == "Dropout", f"D1 should remain Dropout, got {d1_type}"

    # D2: resid_dropout 应替换为 AGDDropoutWrapper
    d2_type = type(model.transformer.h[0].attn.resid_dropout).__name__
    print(f"    D2: attn.resid_dropout  = {d2_type}")
    assert d2_type == "AGDDropoutWrapper", f"D2 should be AGDDropoutWrapper, got {d2_type}"

    # D3: mlp.dropout 应替换为 AGDDropoutWrapper
    d3_type = type(model.transformer.h[0].mlp.dropout).__name__
    print(f"    D3: mlp.dropout         = {d3_type}")
    assert d3_type == "AGDDropoutWrapper", f"D3 should be AGDDropoutWrapper, got {d3_type}"

    # D4: transformer.drop 应替换为 AGDDropoutWrapper
    d4_type = type(model.transformer.drop).__name__
    print(f"    D4: transformer.drop    = {d4_type}")
    assert d4_type == "AGDDropoutWrapper", f"D4 should be AGDDropoutWrapper, got {d4_type}"

    # 统计 AGDDropoutWrapper 数量: 1 (D4) + n_layer * 2 (D2 + D3 per block)
    agd_count = sum(1 for m in model.modules() if isinstance(m, AGDDropoutWrapper))
    expected = 1 + config.n_layer * 2
    print(f"\n  AGDDropoutWrapper 总数: {agd_count} (预期 {expected})")
    assert agd_count == expected, f"Expected {expected}, got {agd_count}"

    # 检查 upstream_linear 引用
    for i, block in enumerate(model.transformer.h):
        d2 = block.attn.resid_dropout
        d3 = block.mlp.dropout
        assert d2.upstream_linear is block.attn.c_proj, f"D2 L{i} upstream should be attn.c_proj"
        assert d3.upstream_linear is block.mlp.c_proj, f"D3 L{i} upstream should be mlp.c_proj"
    d4 = model.transformer.drop
    assert d4.upstream_linear is None, "D4 upstream should be None"

    print("  ✅ 所有 dropout 位置完全对齐!")
    print()
    return model, generator, agd_config


def test_random_dropout_alignment():
    """测试 Random Dropout 也正确替换"""
    print("=" * 50)
    print("🧪 Test 3: Dropout 位置对齐 (Random 模式)")
    print("=" * 50)

    config = GPT2Config(
        n_embd=64, n_head=2, n_layer=2,
        n_inner=128, vocab_size=1000, n_positions=128,
    )
    model = GPT2LMHeadModel(config)

    random_config = {'mode': 'random', 'dropout_p': 0.1}
    model, generator = inject_dropout_to_gpt2(model, random_config)
    assert generator is None

    # 所有 3 个位置都应该替换为 RandomDropoutWrapper
    d2_type = type(model.transformer.h[0].attn.resid_dropout).__name__
    d3_type = type(model.transformer.h[0].mlp.dropout).__name__
    d4_type = type(model.transformer.drop).__name__
    print(f"  D2: {d2_type}, D3: {d3_type}, D4: {d4_type}")

    assert d2_type == "RandomDropoutWrapper"
    assert d3_type == "RandomDropoutWrapper"
    assert d4_type == "RandomDropoutWrapper"

    # 测试前向传播
    model.train()
    input_ids = torch.randint(0, 1000, (2, 32))
    labels = torch.randint(0, 1000, (2, 32))
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print("  ✅ Random Dropout 对齐正确!")
    print()


def test_forward_backward(model, generator, agd_config):
    """测试前向 + 反向传播"""
    print("=" * 50)
    print("🧪 Test 4: 前向 + 反向传播 (Phase A & B)")
    print("=" * 50)

    device = 'cpu'
    model = model.to(device)
    generator = generator.to(device)
    model.train()

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    batch = {'input_ids': input_ids, 'labels': labels}

    model_params = [p for p in model.parameters()]
    gen_params = list(generator.parameters())

    # --- Phase A: Generator 训练 ---
    print("  Phase A: Generator 训练...")
    for p in model_params:
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, AGDDropoutWrapper):
            m.phase_a = True

    opt_gen = AdamW(gen_params, lr=3e-4)
    opt_gen.zero_grad()

    outputs = model(**batch)
    task_loss = outputs.loss
    print(f"    Task loss: {task_loss.item():.4f}")

    gen_loss, cost_val = compute_gen_loss(model, task_loss, agd_config)
    print(f"    Gen loss: {gen_loss.item():.4f} | Cost: {cost_val:.4f}")

    gen_loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gen_params)
    print(f"    Generator 有非零梯度: {has_grad}")
    assert has_grad, "Generator should receive gradients in Phase A"

    opt_gen.step()
    opt_gen.zero_grad()

    for m in model.modules():
        if isinstance(m, AGDDropoutWrapper):
            m.stats = {'mask': None, 'probs': None, 'logits': None}
            m.phase_a = False
    for p in model_params:
        p.requires_grad = True
    print("  ✅ Phase A 完成!")

    # --- Phase B: Model 训练 ---
    print("  Phase B: Model 训练...")
    opt_model = AdamW(model_params, lr=1e-4)
    opt_model.zero_grad()

    outputs = model(**batch)
    loss_model = outputs.loss
    print(f"    Model loss: {loss_model.item():.4f}")

    loss_model.backward()

    has_model_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in model_params if p.requires_grad)
    print(f"    Model 有非零梯度: {has_model_grad}")
    assert has_model_grad, "Model should receive gradients in Phase B"

    # 测试 grad_ema 更新
    for m in model.modules():
        if isinstance(m, AGDDropoutWrapper) and m.upstream_linear is not None:
            grad_signal = _get_grad_per_output(m.upstream_linear)
            if grad_signal is not None:
                m.update_grad_ema(grad_signal)
                print(f"    {m.name} grad_ema shape: {m.grad_ema.shape}, mean: {m.grad_ema.mean():.6f}")

    opt_model.step()
    opt_model.zero_grad()
    for m in model.modules():
        if isinstance(m, AGDDropoutWrapper):
            m.stats = {'mask': None, 'probs': None, 'logits': None}

    print("  ✅ Phase B 完成!")
    print()


def test_full_training_loop():
    """测试完整的微型训练循环 (3 步)"""
    print("=" * 50)
    print("🧪 Test 5: 完整训练循环 (3 步)")
    print("=" * 50)

    device = 'cpu'

    config = GPT2Config(
        n_embd=64, n_head=2, n_layer=2,
        n_inner=128, vocab_size=1000, n_positions=128,
        resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)

    agd_config = {
        'mode': 'agd', 'dropout_p': 0.1,
        'gen_steps': 2, 'drop_cost_base': 0.1, 'drop_limit': 0.1,
        'limit_penalty': 7.0, 'entropy_weight': 0.1, 'task_loss_weight': 0.2,
        'grad_accum': 1,
    }

    model, generator = inject_dropout_to_gpt2(model, agd_config)
    model = model.to(device)
    generator = generator.to(device)
    model.train()

    model_params = list(model.parameters())
    gen_params = list(generator.parameters())
    opt_model = AdamW(model_params, lr=1e-4)
    opt_gen = AdamW(gen_params, lr=3e-4)

    losses = []
    for step in range(3):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        batch = {'input_ids': input_ids, 'labels': labels}

        # Phase A
        for p in model_params:
            p.requires_grad = False
        for m in model.modules():
            if isinstance(m, AGDDropoutWrapper):
                m.phase_a = True

        for gs in range(agd_config['gen_steps']):
            opt_gen.zero_grad()
            outputs = model(**batch)
            gen_loss, cost_val = compute_gen_loss(model, outputs.loss, agd_config)
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
            opt_gen.step()
            opt_gen.zero_grad()
            for m in model.modules():
                if isinstance(m, AGDDropoutWrapper):
                    m.stats = {'mask': None, 'probs': None, 'logits': None}

        for m in model.modules():
            if isinstance(m, AGDDropoutWrapper):
                m.phase_a = False
        for p in model_params:
            p.requires_grad = True

        # Phase B
        opt_model.zero_grad()
        outputs = model(**batch)
        loss_model = outputs.loss
        loss_model.backward()

        # 更新 grad_ema
        for m in model.modules():
            if isinstance(m, AGDDropoutWrapper) and m.upstream_linear is not None:
                grad_signal = _get_grad_per_output(m.upstream_linear)
                if grad_signal is not None:
                    m.update_grad_ema(grad_signal / agd_config['grad_accum'])

        torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
        opt_model.step()
        opt_model.zero_grad()

        for m in model.modules():
            if isinstance(m, AGDDropoutWrapper):
                m.stats = {'mask': None, 'probs': None, 'logits': None}

        losses.append(loss_model.item())
        print(f"  Step {step+1}: loss={loss_model.item():.4f}, gen_loss={gen_loss.item():.4f}, cost={cost_val:.4f}")

    print(f"  Loss 趋势: {' → '.join(f'{l:.4f}' for l in losses)}")
    print("  ✅ 完整训练循环通过!")
    print()


def test_with_real_data():
    """测试用真实 .bin 数据"""
    print("=" * 50)
    print("🧪 Test 6: 真实 .bin 数据加载")
    print("=" * 50)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path):
        print("  ⚠️ 没有找到 data/train.bin，创建临时测试数据...")
        create_tiny_data(train_path, num_tokens=4096)
        create_tiny_data(val_path, num_tokens=1024)

    block_size = 128
    dataset = PretrainDataset(train_path, block_size=block_size)
    collator = DataCollatorForLM()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collator, drop_last=True
    )

    batch = next(iter(loader))
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    assert batch['input_ids'].shape == (2, block_size)
    print("  ✅ 数据加载正常!")
    print()


def main():
    print("\n🚀 AGD GPT-2 端到端流程验证\n")

    test_conv1d_helpers()
    model, generator, agd_config = test_dropout_alignment()
    test_random_dropout_alignment()
    test_forward_backward(model, generator, agd_config)
    test_full_training_loop()
    test_with_real_data()

    print("=" * 50)
    print("🎉 所有测试通过！AGD GPT-2 实现正确！")
    print("=" * 50)


if __name__ == "__main__":
    main()
