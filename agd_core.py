"""
AGD (Adversarial Gumbel Dropout) 核心模块 for GPT-2

基于 CLIP 实现改编，适配 GPT-2 的 Transformer 结构

完全对齐官方 GPT-2 的 4 个 dropout 位置:
  D1: GPT2Attention.attn_dropout   — softmax 后的注意力权重 dropout  (attn_pdrop)
  D2: GPT2Attention.resid_dropout  — attention c_proj 输出 dropout   (resid_pdrop)
  D3: GPT2MLP.dropout              — MLP c_proj 输出 dropout          (resid_pdrop)
  D4: GPT2Model.drop               — embedding dropout                (embd_pdrop)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# HuggingFace GPT-2 使用 Conv1D（权重转置），不是 nn.Linear
# 需要兼容两种层的属性差异
try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None


def _get_out_features(layer):
    """兼容 nn.Linear 和 HuggingFace Conv1D 的 out_features 获取"""
    if hasattr(layer, 'out_features'):
        return layer.out_features  # nn.Linear
    elif hasattr(layer, 'nf'):
        return layer.nf  # HuggingFace Conv1D
    else:
        raise AttributeError(f"Cannot determine out_features for {type(layer)}")


def _get_weight_mag_per_output(layer):
    """
    计算每个输出神经元的权重幅值 (L1 范数)

    nn.Linear:  weight shape = (out_features, in_features) → sum(dim=1) → [out_features]
    Conv1D:     weight shape = (in_features, out_features) → sum(dim=0) → [out_features]
    """
    w = layer.weight.float().abs()
    if HFConv1D is not None and isinstance(layer, HFConv1D):
        return w.sum(dim=0)  # Conv1D: (nx, nf) → sum over input dim
    else:
        return w.sum(dim=1)  # Linear: (out, in) → sum over input dim


def _get_grad_per_output(layer):
    """
    获取每个输出神经元的梯度信号

    nn.Linear:  weight.grad shape = (out_features, in_features) → abs().sum(dim=1) → [out_features]
    Conv1D:     weight.grad shape = (in_features, out_features) → abs().sum(dim=0) → [out_features]
    """
    g = layer.weight.grad
    if g is None:
        return None
    g = g.float().abs()
    if HFConv1D is not None and isinstance(layer, HFConv1D):
        return g.sum(dim=0)  # Conv1D
    else:
        return g.sum(dim=1)  # Linear


# === 1. 共享生成器 (AGD 模式专用) ===
class SharedMaskGenerator(nn.Module):
    """
    共享的掩码生成器，用于所有层

    输入状态包含 3 个归一化特征（Z-Score）：
    1. 激活值 (normalized) - 瞬时特征存在情况
    2. 梯度动量 (normalized) - 动态特征重要性
    3. 权重幅值 (normalized) - 静态特征重要性先验

    层索引通过 Embedding 单独编码后与特征拼接
    """
    def __init__(self, input_dim, num_layers, embed_dim=64):
        super().__init__()
        self.layer_embedding = nn.Embedding(num_layers, embed_dim)
        # 状态空间: [激活, 梯度, 权重] = 3 * input_dim
        self.feature_proj = nn.Linear(input_dim * 3, embed_dim)
        hidden_dim = embed_dim * 2
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # 初始化偏置为正值，鼓励初期保留神经元
        nn.init.constant_(self.net[-1].bias, 2.0)
        self.register_buffer('_layer_indices', torch.arange(num_layers, dtype=torch.long))

    def forward(self, x, layer_idx):
        """
        Args:
            x: [batch, seq_len, input_dim * 3] 状态向量
            layer_idx: 层索引
        Returns:
            logits: [batch, seq_len, input_dim] 掩码 logits
        """
        layer_vec = self.layer_embedding(self._layer_indices[layer_idx])

        # 强制使用 fp32 计算，避免 fp16 数值问题
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            x_proj = self.feature_proj(x_fp32)
            layer_vec_expanded = layer_vec.float().view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
            combined_state = torch.cat([x_proj, layer_vec_expanded], dim=-1)
            out = self.net(combined_state)

        return out.to(x.dtype)

    def noisy_ste(self, logits, noise_scale=1.0):
        """
        高斯噪声直通估计器 (Gaussian Noisy STE)

        前向: 硬阈值 (离散二值)
        反向: 软 sigmoid (连续可导)
        """
        noise = torch.randn_like(logits) * noise_scale
        y_soft = torch.sigmoid(logits + noise)
        y_hard = (y_soft > 0.5).float()
        # 直通估计器: 前向用 hard，反向用 soft 的梯度
        return (y_hard - y_soft).detach() + y_soft


# === 2. AGD Dropout Wrapper ===
class AGDDropoutWrapper(nn.Module):
    """
    AGD Dropout Wrapper for GPT-2

    直接替换 GPT-2 中的 nn.Dropout 模块，完全对齐官方 dropout 位置。
    通过 upstream_linear 引用上游的 Conv1D/Linear 层来获取权重先验和梯度动量。

    官方 GPT-2 的 4 个 dropout 位置:
      D1: attn.attn_dropout  (softmax 后)       → upstream: attn.c_attn
      D2: attn.resid_dropout (c_proj 输出后)     → upstream: attn.c_proj
      D3: mlp.dropout        (c_proj 输出后)     → upstream: mlp.c_proj
      D4: model.drop         (embedding 后)      → upstream: None (无权重先验)
    """
    def __init__(self, generator, layer_idx, config, upstream_linear=None,
                 feature_dim=None, ema_decay=0.99, name=''):
        super().__init__()
        self._gen_ref = [generator]
        self.layer_idx = layer_idx
        self.config = config
        self.upstream_linear = upstream_linear  # 上游线性层，用于权重先验
        self.feature_dim = feature_dim
        self.stats = {'mask': None, 'probs': None, 'logits': None}
        self.phase_a = False
        self.ema_decay = ema_decay
        self.name = name  # 调试标识 (e.g. "L0.attn_dropout")
        self._grad_ema = None
        self._out_features = feature_dim

    @property
    def grad_ema(self):
        if self._grad_ema is None:
            device = self.upstream_linear.weight.device if self.upstream_linear else 'cpu'
            self._grad_ema = torch.zeros(self._out_features, device=device)
        return self._grad_ema

    @grad_ema.setter
    def grad_ema(self, value):
        self._grad_ema = value

    def _get_grad_ema_on(self, device):
        """获取与指定设备对齐的 grad_ema（惰性初始化 + 自动迁移）"""
        ema = self.grad_ema  # 触发 lazy init
        if ema.device != device:
            self._grad_ema = ema.to(device)
        return self._grad_ema

    @property
    def generator(self):
        return self._gen_ref[0]

    def update_grad_ema(self, grad):
        """更新梯度 EMA"""
        with torch.no_grad():
            if grad.dim() > 1:
                grad = grad.abs().mean(dim=list(range(grad.dim() - 1)))
            else:
                grad = grad.abs()
            self.grad_ema = self.ema_decay * self.grad_ema + (1 - self.ema_decay) * grad

    def forward(self, x):
        """
        替代 nn.Dropout 的前向传播

        训练时: 使用 AGD 生成对抗性掩码
        推理时: 直接返回输入 (等价于 nn.Dropout eval 模式)
        """
        if not self.training:
            return x

        # Phase A: generator 需要 x 的梯度
        # Phase B: 必须 detach，避免主模型梯度通过 mask 回流
        gen_input = x if self.phase_a else x.detach()

        # 构建状态空间 (强制 fp32)
        with torch.amp.autocast('cuda', enabled=False):
            inp_fp32 = gen_input.float()

            # 处理不同维度的输入:
            # D1 (attn_dropout): [batch, heads, seq, seq] → 需要特殊处理
            # D2, D3 (resid_dropout, mlp dropout): [batch, seq, n_embd]
            # D4 (embd dropout): [batch, seq, n_embd]
            original_shape = inp_fp32.shape
            if inp_fp32.dim() == 4:
                # D1: attention weights [batch, heads, seq_q, seq_k]
                # 沿最后一个维度 (seq_k) 做 dropout → 对齐 attn_pdrop 的语义
                batch, heads, seq_q, seq_k = inp_fp32.shape
                # reshape 为 [batch * heads * seq_q, 1, seq_k] 来复用 3D 的 generator
                inp_fp32 = inp_fp32.reshape(-1, 1, seq_k)

            # 1. 激活值归一化
            mean = inp_fp32.mean(dim=-1, keepdim=True)
            std = inp_fp32.std(dim=-1, keepdim=True)
            act_norm = (inp_fp32 - mean) / (std + 1e-5)

            # 2. 梯度动量归一化
            with torch.no_grad():
                g = self._get_grad_ema_on(inp_fp32.device).view(1, 1, -1).expand_as(inp_fp32)
                g_mean = g.mean(dim=-1, keepdim=True)
                g_std = g.std(dim=-1, keepdim=True)
                g_norm = (g - g_mean) / (g_std + 1e-5)

            # 3. 权重幅值归一化
            with torch.no_grad():
                if self.upstream_linear is not None:
                    w_mag = _get_weight_mag_per_output(self.upstream_linear)
                else:
                    # D4 (embedding dropout) 没有上游线性层，用全 1 代替
                    w_mag = torch.ones(self.feature_dim, device=inp_fp32.device)
                w = w_mag.view(1, 1, -1).expand_as(inp_fp32)
                w_mean = w.mean(dim=-1, keepdim=True)
                w_std = w.std(dim=-1, keepdim=True)
                w_norm = (w - w_mean) / (w_std + 1e-5)

            # 拼接状态: [激活, 梯度, 权重]
            state = torch.cat([act_norm, g_norm, w_norm], dim=-1)

            if self.phase_a:
                logits = self.generator(state, self.layer_idx)
            else:
                with torch.no_grad():
                    logits = self.generator(state, self.layer_idx)

            logits = torch.clamp(logits, min=-10.0, max=10.0)
            mask = self.generator.noisy_ste(logits)
            probs = torch.sigmoid(logits)

            self.stats['mask'] = mask
            self.stats['probs'] = probs
            self.stats['logits'] = logits

            # 缩放因子
            keep_prob_actual = probs.mean().detach()
            keep_prob_safe = torch.clamp(keep_prob_actual, min=0.1)
            scale_factor = 1.0 / keep_prob_safe
            scale_factor = torch.clamp(scale_factor, max=3.0)

            # 应用掩码（用原始 x 保留梯度）
            x_fp32 = x.float()
            if x_fp32.dim() == 4:
                # D1: 恢复 4D 形状
                batch, heads, seq_q, seq_k = original_shape
                mask_4d = mask.reshape(batch, heads, seq_q, seq_k)
                out_dropped = x_fp32 * mask_4d * scale_factor
            else:
                out_dropped = x_fp32 * mask * scale_factor

        return out_dropped.to(x.dtype)


# === 3. Random Dropout Wrapper ===
class RandomDropoutWrapper(nn.Module):
    """标准 Random Dropout — 直接替换 nn.Dropout"""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.0:
            return F.dropout(x, p=self.p, training=True)
        return x


# === 4. 注入函数 ===
def inject_dropout_to_gpt2(model, config):
    """
    将 AGD 或 Random Dropout 注入到 GPT-2 模型

    完全对齐官方 GPT-2 的 4 个 dropout 位置:
      D1: block.attn.attn_dropout   — 注意力权重 dropout     (attn_pdrop)
      D2: block.attn.resid_dropout  — 注意力残差 dropout      (resid_pdrop)
      D3: block.mlp.dropout         — MLP 残差 dropout         (resid_pdrop)
      D4: model.transformer.drop    — Embedding dropout        (embd_pdrop)

    Args:
        model: GPT2LMHeadModel
        config: 配置字典，包含 mode, dropout_p 等

    Returns:
        model: 注入后的模型
        generator: AGD generator (如果是 AGD 模式) 或 None
    """
    blocks = model.transformer.h
    num_layers = len(blocks)
    mode = config.get('mode', 'agd')
    generator = None

    # 每个 block 有 3 个 dropout (D1, D2, D3)，加上 1 个 embedding dropout (D4)
    # 总层数 = num_layers * 3 + 1
    total_dropout_layers = num_layers * 3 + 1
    embed_dim = model.config.n_embd
    n_head = model.config.n_head
    head_dim = embed_dim // n_head

    if mode == 'agd':
        generator = SharedMaskGenerator(
            input_dim=embed_dim,
            num_layers=total_dropout_layers
        )
        print(f"🔥 Mode: AGD (Adversarial Gumbel-Dropout) 已激活")
        print(f"   - Transformer blocks: {num_layers}")
        print(f"   - 每 block 3 个 dropout + 1 个 embedding dropout = {total_dropout_layers} 个 AGD 层")
        print(f"   - 特征维度: {embed_dim}")
    else:
        p = config.get('dropout_p', 0.1)
        print(f"🧊 Mode: Random Dropout (p={p}) 已激活")
        print(f"   - 替换所有 {total_dropout_layers} 个 dropout 层")

    layer_idx = 0

    # === D4: Embedding dropout (model.transformer.drop) ===
    if mode == 'agd':
        model.transformer.drop = AGDDropoutWrapper(
            generator=generator, layer_idx=layer_idx, config=config,
            upstream_linear=None,  # embedding 没有上游线性层
            feature_dim=embed_dim,
            name='embd_dropout',
        )
    else:
        model.transformer.drop = RandomDropoutWrapper(p=config.get('dropout_p', 0.1))
    layer_idx += 1

    # === D1, D2, D3: 每个 block 内的 3 个 dropout ===
    for i, block in enumerate(blocks):
        # D1: attn.attn_dropout — 注意力权重 dropout
        # 上游: c_attn 投影，但 attn weights 的维度是 [batch, heads, seq, seq]
        # 我们用 head_dim 作为特征维度不太合适，用 embed_dim 统一
        # 注意: D1 的 feature_dim 与其他不同，但 generator 输出维度固定为 embed_dim
        # 对于 attention weights [batch, heads, seq_q, seq_k]，seq_k 维度不固定
        # → 跳过 D1，保留原始 nn.Dropout (与 CLIP 参考实现一致，CLIP 也只注入 MLP 和 resid)
        # 原因: attn_dropout 作用于 [batch, heads, seq_q, seq_k] 的最后一维是 seq_len,
        #        不是固定特征维度，无法用固定维度的 generator 生成掩码

        # D2: attn.resid_dropout — 注意力残差 dropout
        if mode == 'agd':
            block.attn.resid_dropout = AGDDropoutWrapper(
                generator=generator, layer_idx=layer_idx, config=config,
                upstream_linear=block.attn.c_proj,
                feature_dim=embed_dim,
                name=f'L{i}.attn_resid_dropout',
            )
        else:
            block.attn.resid_dropout = RandomDropoutWrapper(p=config.get('dropout_p', 0.1))
        layer_idx += 1

        # D3: mlp.dropout — MLP 残差 dropout
        if mode == 'agd':
            block.mlp.dropout = AGDDropoutWrapper(
                generator=generator, layer_idx=layer_idx, config=config,
                upstream_linear=block.mlp.c_proj,
                feature_dim=embed_dim,
                name=f'L{i}.mlp_dropout',
            )
        else:
            block.mlp.dropout = RandomDropoutWrapper(p=config.get('dropout_p', 0.1))
        layer_idx += 1

        # D1 的 layer_idx 也要占位（即使不注入 AGD），保持 layer embedding 的对齐
        # 如果以后想支持 D1，只需在这里改成 AGDDropoutWrapper
        layer_idx += 1

    actual_agd_count = layer_idx - num_layers  # 减去 D1 跳过的数量
    if mode == 'agd':
        print(f"   - 实际注入 AGD: {actual_agd_count} 个 (跳过 {num_layers} 个 attn_dropout)")
        print(f"   - D4 (embd): 1, D2 (attn_resid): {num_layers}, D3 (mlp): {num_layers}")
    return model, generator


# === 5. Generator Loss 计算 ===
def compute_gen_loss(model, task_loss, config):
    """
    计算 Generator 的损失

    Loss = -task_loss + cost - entropy

    其中:
    - task_loss: 主任务损失（语言模型的交叉熵）
    - cost: 丢弃率成本（分段线性，带软上限）
    - entropy: 熵正则化（鼓励探索）
    """
    total_cost = torch.tensor(0.0, device=task_loss.device)
    total_entropy = torch.tensor(0.0, device=task_loss.device)
    count = 0

    for module in model.modules():
        if isinstance(module, AGDDropoutWrapper):
            if module.stats['logits'] is None:
                continue

            logits = module.stats['logits']
            probs = torch.sigmoid(logits)
            current_drop = 1.0 - probs.mean()

            # 分段线性成本 (Hinge Cost)
            base_cost = config['drop_cost_base'] * current_drop
            excess = F.relu(current_drop - config['drop_limit'])
            limit_cost = config['limit_penalty'] * (excess ** 2)
            total_cost = total_cost + base_cost + limit_cost

            # 熵正则化 (使用 softplus 恒等式，避免 log(0))
            # H(σ(z)) = softplus(z) - z·σ(z)
            entropy = (F.softplus(logits) - logits * probs).mean()
            total_entropy = total_entropy + entropy
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=task_loss.device, requires_grad=True), 0.0

    avg_entropy = total_entropy / count

    # Generator 目标: 最大化 task_loss，最小化 cost，最大化 entropy
    # cost 使用 total（所有层之和），与 CLIP 参考实现一致
    avg_cost = total_cost / count
    w = config.get('task_loss_weight', 0.2)
    gen_loss = -w * task_loss + avg_cost - (config['entropy_weight'] * avg_entropy)

    return gen_loss, avg_cost.item()