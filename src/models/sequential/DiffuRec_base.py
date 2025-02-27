import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import SequentialModel

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps,
                                  lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: 1 - np.sqrt(t + 0.0001))
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps,
                                       lambda t: np.cos((t + 0.1) / 1.1 * math.pi / 2) ** 2)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def random_schedule_sampler(num_timesteps, batch_size, device):
    t = torch.randint(low=0, high=num_timesteps, size=(batch_size,), device=device)
    # weights 通常用于加权 loss，这里可直接返回全 1
    weights = torch.ones_like(t, dtype=torch.float, device=device)
    return t, weights

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        x = self.w_1(hidden)
        x = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        x = self.w_2(self.dropout(x))
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)
        for linear in self.linear_layers:
            nn.init.xavier_normal_(linear.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
                   for l, x in zip(self.linear_layers, (q, k, v))]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        hidden = torch.matmul(attn, v)
        hidden = hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head)
        hidden = self.w_layer(hidden)
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden_size, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)
        self.input_sublayer = SublayerConnection(hidden_size, dropout)
        self.output_sublayer = SublayerConnection(hidden_size, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden,
                                     lambda _x: self.attention(_x, _x, _x, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, hidden_size, num_blocks, dropout, heads=4):
        super(Transformer_rep, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, heads, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, hidden, mask):
        for block in self.blocks:
            hidden = block(hidden, mask)
        return hidden

def space_timesteps(num_timesteps):
    return list(range(num_timesteps))

class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, num_blocks, dropout, heads, lambda_uncertainty):
        super(Diffu_xstart, self).__init__()
        self.hidden_size = hidden_size
        self.lambda_uncertainty = lambda_uncertainty

        self.att_transformer = Transformer_rep(hidden_size, num_blocks, dropout, heads)
        self.norm_diffu_rep = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.time_embed_dim = hidden_size * 4
        self.linear_t1 = nn.Linear(hidden_size, self.time_embed_dim)
        self.linear_t2 = nn.Linear(self.time_embed_dim, hidden_size)
        self.silu = SiLU()

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, seq_emb, x_t, t, mask_seq):
        """
        seq_emb: [B, L, D]  原始序列的嵌入
        x_t:     [B, D]     当前扩散时刻的噪声表示
        t:       [B]        当前时间步
        mask_seq:[B, L]     序列有效位置
        """
        t_emb = self.timestep_embedding(t, self.hidden_size)
        t_emb = self.silu(self.linear_t1(t_emb))
        t_emb = self.linear_t2(t_emb)

        lambda_noise = torch.normal(mean=torch.full_like(seq_emb, self.lambda_uncertainty),
                                    std=torch.full_like(seq_emb, self.lambda_uncertainty)).to(seq_emb.device)
        # 将 x_t.unsqueeze(1) broadcast 到 [B, L, D]
        x_t_expand = x_t.unsqueeze(1).expand_as(seq_emb)

        # 融合: seq_emb + lambda_noise * x_t_expand + (optionally) t_emb
        fusion = seq_emb + lambda_noise * x_t_expand

        # 过 Transformer
        seq_rep = self.att_transformer(fusion, mask_seq)
        seq_rep = self.norm_diffu_rep(self.dropout(seq_rep))
        # 取序列最后一个位置或平均池化
        # 以最后一个非零位置 embedding 作为整体表示
        # mask_seq: [B, L], lengths = mask_seq.sum(dim=1)
        lengths = mask_seq.sum(dim=1).long()
        batch_idx = torch.arange(seq_emb.size(0), device=seq_emb.device)
        out = seq_rep[batch_idx, lengths - 1, :]  # [B, D]

        return out


class DiffusionCore(nn.Module):
    def __init__(self, hidden_size, num_blocks, dropout, heads,
                 diffusion_steps, noise_schedule, lambda_uncertainty, rescale_timesteps):
        super(DiffusionCore, self).__init__()
        self.hidden_size = hidden_size
        self.num_timesteps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.rescale_timesteps = rescale_timesteps

        betas = get_named_beta_schedule(self.noise_schedule, self.num_timesteps)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.use_timesteps = space_timesteps(self.num_timesteps)

        self.xstart_model = Diffu_xstart(hidden_size, num_blocks, dropout, heads, lambda_uncertainty)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t.float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def q_posterior_mean(self, x_start, x_t, t):
        coef1 = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
        return coef1 * x_start + coef2 * x_t

    def p_mean_variance(self, seq_emb, x_t, t, mask_seq):
        x_0 = self.xstart_model(seq_emb, x_t, self._scale_timesteps(t), mask_seq)  # [B, D]
        model_mean = self.q_posterior_mean(x_0, x_t, t)
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        return model_mean, model_log_variance, x_0

    def p_sample(self, seq_emb, x_t, t, mask_seq):
        model_mean, model_log_variance, x_0 = self.p_mean_variance(seq_emb, x_t, t, mask_seq)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        x_prev = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return x_prev, x_0

    def forward_diffusion(self, seq_emb, x_start, mask_seq):
        """
        正向扩散，用于训练时随机采样 t 并得到 x_t，再调用 xstart_model 预测 x_0。
        """
        batch_size = x_start.size(0)
        device = x_start.device
        # 随机采样 t
        t, weights = random_schedule_sampler(self.num_timesteps, batch_size, device)
        # 从 x_0 得到 x_t
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        # 用 x_t 和 seq_emb 预测 x_0
        x_0 = self.xstart_model(seq_emb, x_t, self._scale_timesteps(t), mask_seq)
        return x_0, x_t, t, weights

    def reverse_diffusion(self, seq_emb, x_T, mask_seq):
        """
        逆向扩散，用于测试时从随机噪声 x_T 逆推到 x_0。
        """
        device = x_T.device
        x_prev = x_T
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i] * x_T.size(0), device=device, dtype=torch.long)
            with torch.no_grad():  # 禁用梯度追踪以节省内存
                x_prev, _ = self.p_sample(seq_emb, x_prev, t, mask_seq)
            del t  # 显式释放内存
        return x_prev


class DiffuRec_baseBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of model')     # 源模型默认128
        parser.add_argument('--num_blocks', type=int, default=2, help='Number of Transformer blocks')   # 源模型默认4
        parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Sampler name (unused in this code, kept for compatibility)')
        parser.add_argument('--diffusion_steps', type=int, default=16, help='Number of diffusion steps')  # 源模型默认32
        parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='Uncertainty weight')
        parser.add_argument('--noise_schedule', type=str, default='trunc_lin', help='Noise schedule')
        parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout for item embedding')
        parser.add_argument('--rescale_timesteps', type=bool, default=True, help='Whether to rescale timesteps')
        parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in Transformer')
        return parser

    def _base_init(self, args, corpus):
        self.hidden_size = args.hidden_size
        self.num_blocks = args.num_blocks
        self.diffusion_steps = args.diffusion_steps
        self.lambda_uncertainty = args.lambda_uncertainty
        self.noise_schedule = args.noise_schedule
        self.emb_dropout = args.emb_dropout
        self.rescale_timesteps = args.rescale_timesteps
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        self.item_num = corpus.n_items + 1
        self.device = args.device

        self.item_embeddings = nn.Embedding(self.item_num, self.hidden_size)
        self.position_embeddings = nn.Embedding(50, self.hidden_size)
        self.embed_dropout = nn.Dropout(self.emb_dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)

        self.diffusion_core = DiffusionCore(
            hidden_size=self.hidden_size,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            heads=self.num_heads,
            diffusion_steps=self.diffusion_steps,
            noise_schedule=self.noise_schedule,
            lambda_uncertainty=self.lambda_uncertainty,
            rescale_timesteps=self.rescale_timesteps
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_normal_(m.weight)

    def loss(self, out_dict):
        prediction = out_dict['prediction']  # [B, 100]，第一列为正样本，其余为负样本
        if prediction.dim() == 2 and prediction.size(1) == 1000:  # 如果包含 100 个候选（1 正 + 99 负）
            pos_scores = prediction[:, 0]  # 正样本分数 [B]
            neg_scores = prediction[:, 1:]  # 负样本分数 [B, 99]
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores.mean(dim=-1)))  # [B]
            return loss.mean()
        else:
            labels = out_dict.get('labels', torch.zeros_like(prediction.squeeze(-1), dtype=torch.long))
            return nn.CrossEntropyLoss()(prediction, labels)

    def forward(self, feed_dict):
        """
        feed_dict 包含:
          'history_items': [B, L] 历史序列
          'item_id': [B, 100] 候选物品（1 个正样本和 99 个负样本）
        """
        sequence = feed_dict['history_items']  # [B, L]
        tags = feed_dict['item_id']  # [B, 100]，1 个正样本和 99 个负样本

        mask_seq = (sequence > 0).float()  # [B, L]
        seq_emb = self.item_embeddings(sequence)  # [B, L, D]
        seq_emb = self.embed_dropout(seq_emb)
        seq_emb = self.layer_norm(seq_emb)

        pos_tags = tags[:, 0]  # 取正样本 [B]
        pos_tag_emb = self.item_embeddings(pos_tags)  # [B, D]
        neg_tags = tags[:, 1:] if tags.size(1) > 1 else None  # 取负样本 [B, 99]

        predictions = []

        if self.training:
            x_0, x_t, t, weights = self.diffusion_core.forward_diffusion(seq_emb, pos_tag_emb, mask_seq)
            pos_prediction = (x_0 * pos_tag_emb).sum(dim=-1, keepdim=True)  # [B, 1]

            if neg_tags is not None:
                neg_tag_embs = self.item_embeddings(neg_tags)  # [B, 99, D]
                neg_prediction = (x_0.unsqueeze(1) * neg_tag_embs).sum(dim=-1)  # [B, 99]
                prediction = torch.cat([pos_prediction.squeeze(-1).unsqueeze(-1), neg_prediction], dim=-1)  # [B, 100]
            else:
                prediction = pos_prediction  # [B, 1]
        else:
            # 测试推理时，从随机噪声 x_T 逆向还原
            x_T = torch.randn_like(pos_tag_emb)  # [B, D]
            x_0 = self.diffusion_core.reverse_diffusion(seq_emb, x_T, mask_seq)
            pos_prediction = (x_0 * pos_tag_emb).sum(dim=-1, keepdim=True)  # [B, 1]
            if neg_tags is not None:
                neg_tag_embs = self.item_embeddings(neg_tags)  # [B, 99, D]
                neg_prediction = (x_0.unsqueeze(1) * neg_tag_embs).sum(dim=-1)  # [B, 99]
                prediction = torch.cat([pos_prediction.squeeze(-1).unsqueeze(-1), neg_prediction], dim=-1)  # [B, 100]
            else:
                prediction = pos_prediction  # [B, 1]（如果无负样本）

        return {
            'prediction': prediction,
            'rep_diffu': x_0,
            'weights': weights if self.training else None,
            't': t if self.training else None,
            'labels': torch.zeros_like(tags[:, 0], dtype=torch.long) if self.training else None
        }


class DiffuRec_base(SequentialModel, DiffuRec_baseBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'hidden_size', 'num_blocks', 'diffusion_steps',
        'lambda_uncertainty', 'noise_schedule', 'emb_dropout',
        'rescale_timesteps', 'num_heads'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = DiffuRec_baseBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return DiffuRec_baseBase.forward(self, feed_dict)

    def loss(self, out_dict):
        return DiffuRec_baseBase.loss(self, out_dict)

# Impression模式，模仿写的，不知道能不能用
# class DiffuRec_baseImpression(ImpressionSeqModel, DiffuRec_baseBase):
#     reader = 'ImpressionSeqReader'
#     runner = 'ImpressionRunner'
#     extra_log_args = DiffuRec_base.extra_log_args
#
#     @staticmethod
#     def parse_model_args(parser):
#         parser = DiffuRec_baseBase.parse_model_args(parser)
#         return ImpressionSeqModel.parse_model_args(parser)
#
#     def __init__(self, args, corpus):
#         ImpressionSeqModel.__init__(self, args, corpus)
#         self._base_init(args, corpus)
#
#     def forward(self, feed_dict):
#         return DiffuRec_baseBase.forward(self, feed_dict)