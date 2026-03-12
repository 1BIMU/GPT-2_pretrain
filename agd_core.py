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
                 feature_dim=None, ema_decay=0.99, name='', target_p=0.1):
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
        # 使用与 nn.Dropout(p=target_p) 相同的固定 scale factor
        # 确保与 Random Dropout 基线的公平对比
        self.scale_factor = 1.0 / (1.0 - target_p) if target_p < 1.0 else 1.0

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
        """更新梯度 EMA（带 NaN/Inf 保护）"""
        with torch.no_grad():
            if grad.dim() > 1:
                grad = grad.abs().mean(dim=list(range(grad.dim() - 1)))
            else:
                grad = grad.abs()
            # 🛡️ 防止 NaN/Inf 梯度永久污染 grad_ema
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                return  # 跳过本次更新，保留上一次的健康值
            self.grad_ema = self.ema_decay * self.grad_ema + (1 - self.ema_decay) * grad
            # 额外检查：如果 grad_ema 本身被污染，重置为零
            if torch.isnan(self.grad_ema).any() or torch.isinf(self.grad_ema).any():
                self.grad_ema = torch.zeros_like(self.grad_ema)

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
            act_norm = torch.clamp(act_norm, -5.0, 5.0)  # 防止 std≈0 时爆炸

            # 2. 梯度动量归一化（带 NaN 保护）
            with torch.no_grad():
                g = self._get_grad_ema_on(inp_fp32.device).view(1, 1, -1).expand_as(inp_fp32)
                # 🛡️ 如果 grad_ema 含 NaN/Inf，用零替代
                if torch.isnan(g).any() or torch.isinf(g).any():
                    g = torch.zeros_like(g)
                g_mean = g.mean(dim=-1, keepdim=True)
                g_std = g.std(dim=-1, keepdim=True)
                g_norm = (g - g_mean) / (g_std + 1e-5)
                g_norm = torch.clamp(g_norm, -5.0, 5.0)  # 防止爆炸

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
                w_norm = torch.clamp(w_norm, -5.0, 5.0)  # 防止爆炸

            # 拼接状态: [激活, 梯度, 权重]
            state = torch.cat([act_norm, g_norm, w_norm], dim=-1)

            if self.phase_a:
                logits = self.generator(state, self.layer_idx)
            else:
                with torch.no_grad():
                    logits = self.generator(state, self.layer_idx)

            logits = torch.clamp(logits, min=-10.0, max=10.0)
            probs = torch.sigmoid(logits)

            if self.phase_a:
                # Phase A: 使用 noisy_ste 为 generator 提供随机梯度估计
                # Phase A 已临时禁用 gradient checkpointing，噪声不会被重算
                mask = self.generator.noisy_ste(logits)
            else:
                # Phase B: 使用确定性硬掩码（无随机噪声）
                # 这对 gradient checkpointing 至关重要：
                #   gradient checkpointing 会在反向传播时重算每个 block 的前向，
                #   如果 mask 依赖 torch.randn_like()，重算时噪声不同 → mask 不同
                #   → 梯度被错误的 mask 缩放 → 49 层级联放大 → 训练崩溃
                # CLIP 不用 gradient checkpointing，所以没有此问题
                mask = (probs > 0.5).float()

            self.stats['mask'] = mask
            self.stats['probs'] = probs
            self.stats['logits'] = logits

            # 使用固定 scale factor（与 nn.Dropout 的 inverted dropout 一致）
            # 例如 target_p=0.1 → scale = 1/0.9 ≈ 1.1111
            # 这保证了与 Random Dropout 的公平对比，且不会因动态 scale 导致深层累积放大
            sf = self.scale_factor

            # 应用掩码（用原始 x 保留梯度）
            x_fp32 = x.float()
            if x_fp32.dim() == 4:
                # D1: 恢复 4D 形状
                batch, heads, seq_q, seq_k = original_shape
                mask_4d = mask.reshape(batch, heads, seq_q, seq_k)
                out_dropped = x_fp32 * mask_4d * sf
            else:
                out_dropped = x_fp32 * mask * sf

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

    AGD 与 Random Dropout 注入**完全相同的位置** (D2, D3, D4)，
    确保实验对比的公平性。

    使用固定 scale factor = 1/(1-p) 与 nn.Dropout 的 inverted dropout 一致，
    避免动态 scale 在深层残差网络中累积放大。

    D1 (attn_dropout) 因维度不固定 [batch, heads, seq, seq]，两种模式均跳过。

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
    dropout_p = config.get('dropout_p', 0.1)  # 用于 Random 和 AGD 的 scale factor
    generator = None

    # D2 + D3 + D4 = num_layers * 2 + 1
    total_dropout_layers = num_layers * 2 + 1
    embed_dim = model.config.n_embd

    if mode == 'agd':
        generator = SharedMaskGenerator(
            input_dim=embed_dim,
            num_layers=total_dropout_layers
        )
        print(f"🔥 Mode: AGD (Adversarial Gumbel-Dropout) 已激活")
        print(f"   - Transformer blocks: {num_layers}")
        print(f"   - 注入位置: D2 (attn_resid) + D3 (mlp) + D4 (embedding) = {total_dropout_layers} 个")
        print(f"   - 固定 scale factor: 1/(1-{dropout_p}) = {1.0/(1.0-dropout_p):.4f} (与 nn.Dropout 对齐)")
        print(f"   - 特征维度: {embed_dim}")
    else:
        p = config.get('dropout_p', 0.1)
        print(f"🧊 Mode: Random Dropout (p={p}) 已激活")
        print(f"   - 注入位置: D2 (attn_resid) + D3 (mlp) + D4 (embedding) = {total_dropout_layers} 个")

    layer_idx = 0

    # === D4: Embedding dropout (model.transformer.drop) ===
    if mode == 'agd':
        model.transformer.drop = AGDDropoutWrapper(
            generator=generator, layer_idx=layer_idx, config=config,
            upstream_linear=None,  # embedding 没有上游线性层
            feature_dim=embed_dim,
            name='embd_dropout',
            target_p=dropout_p,
        )
        layer_idx += 1
    else:
        model.transformer.drop = RandomDropoutWrapper(p=dropout_p)

    # === D1, D2, D3: 每个 block 内的 dropout ===
    for i, block in enumerate(blocks):
        # D1: attn.attn_dropout — 跳过（维度不固定）
        # 两种模式下均保留原始 nn.Dropout(p=0.0)

        # D2: attn.resid_dropout
        if mode == 'agd':
            block.attn.resid_dropout = AGDDropoutWrapper(
                generator=generator, layer_idx=layer_idx, config=config,
                upstream_linear=block.attn.c_proj,
                feature_dim=embed_dim,
                name=f'L{i}.attn_resid_dropout',
                target_p=dropout_p,
            )
            layer_idx += 1
        else:
            block.attn.resid_dropout = RandomDropoutWrapper(p=dropout_p)

        # D3: mlp.dropout
        if mode == 'agd':
            block.mlp.dropout = AGDDropoutWrapper(
                generator=generator, layer_idx=layer_idx, config=config,
                upstream_linear=block.mlp.c_proj,
                feature_dim=embed_dim,
                name=f'L{i}.mlp_dropout',
                target_p=dropout_p,
            )
            layer_idx += 1
        else:
            block.mlp.dropout = RandomDropoutWrapper(p=dropout_p)

    if mode == 'agd':
        print(f"   - 实际注入 AGD: {layer_idx} 个")
        print(f"     D4 (embd): 1, D2 (attn_resid): {num_layers}, D3 (mlp): {num_layers}")
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
    w = config.get('task_loss_weight', 0.2)
    gen_loss = -w * task_loss + total_cost - (config['entropy_weight'] * avg_entropy)

    return gen_loss, (total_cost / count).item()
