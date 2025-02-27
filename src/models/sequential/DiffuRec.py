import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from models.BaseModel import SequentialModel


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        def alpha_bar(t): return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas, dtype=np.float64)
    elif schedule_name == "sqrt":
        def alpha_bar(t): return 1 - np.sqrt(t + 0.0001)
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas, dtype=np.float64)
    elif schedule_name == "trunc_cos":
        def alpha_bar(t): return np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2
        betas = [min(1 - alpha_bar(0), 0.999)]
        for i in range(num_diffusion_timesteps - 1):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas, dtype=np.float64)
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


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        pass

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([num_timesteps], dtype=np.int32)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= (1 - self.uniform_prob)
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

    def update_with_local_losses(self, local_ts, local_losses):
        timesteps = local_ts.cpu().numpy()
        losses = local_losses.cpu().numpy()
        self.update_with_all_losses(timesteps, losses)


def create_named_schedule_sampler(name, num_timesteps):
    if name == "uniform":
        return UniformSampler(num_timesteps)
    elif name == "lossaware":
        return LossAwareSampler(num_timesteps)
    else:
        raise NotImplementedError(f"Unknown sampler: {name}")


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
        return self.w_2(self.dropout(x))


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
        return self.w_layer(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden_size, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)
        self.input_sublayer = SublayerConnection(hidden_size, dropout)
        self.output_sublayer = SublayerConnection(hidden_size, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _x: self.attention(_x, _x, _x, mask=mask))
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
        t_emb = self.timestep_embedding(t, self.hidden_size)
        t_emb = self.silu(self.linear_t1(t_emb))
        t_emb = self.linear_t2(t_emb)

        lambda_noise = torch.normal(mean=torch.full_like(seq_emb, self.lambda_uncertainty),
                                    std=torch.full_like(seq_emb, self.lambda_uncertainty)).to(seq_emb.device)
        x_t_expand = x_t.unsqueeze(1).expand_as(seq_emb)
        fusion = seq_emb + lambda_noise * x_t_expand

        seq_rep = self.att_transformer(fusion, mask_seq)
        seq_rep = self.norm_diffu_rep(self.dropout(seq_rep))
        lengths = mask_seq.sum(dim=1).long()
        batch_idx = torch.arange(seq_emb.size(0), device=seq_emb.device)
        out = seq_rep[batch_idx, lengths - 1, :]
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

        self.schedule_sampler = create_named_schedule_sampler('lossaware', self.num_timesteps)
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
        x_0 = self.xstart_model(seq_emb, x_t, self._scale_timesteps(t), mask_seq)
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
        batch_size = x_start.size(0)
        device = x_start.device
        t, weights = self.schedule_sampler.sample(batch_size, device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        x_0 = self.xstart_model(seq_emb, x_t, self._scale_timesteps(t), mask_seq)
        return x_0, x_t, t, weights

    def reverse_diffusion(self, seq_emb, x_T, mask_seq):
        device = x_T.device
        x_prev = x_T
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((x_T.size(0),), i, device=device, dtype=torch.long)
            with torch.no_grad():
                x_prev, _ = self.p_sample(seq_emb, x_prev, t, mask_seq)
        return x_prev


class DiffuRecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of model')
        parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
        parser.add_argument('--diffusion_steps', type=int, default=32, help='Number of diffusion steps')
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
        self.item_num = corpus.n_items
        self.device = args.device

        self.item_embeddings = nn.Embedding(self.item_num + 1, self.hidden_size)
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

    def forward(self, feed_dict):
        sequence = feed_dict['history_items']  # [B, L]
        tags = feed_dict['item_id']            # [B, N] or [B]
        if tags.dim() == 1:
            tags = tags.unsqueeze(1)           # 确保 tags 是 [B, N]
        num_candidates = tags.size(1)

        mask_seq = (sequence > 0).float()      # [B, L]
        seq_emb = self.item_embeddings(sequence)
        seq_emb = self.embed_dropout(seq_emb)
        seq_emb = self.layer_norm(seq_emb)

        candidate_emb = self.item_embeddings(tags)  # [B, N, D]

        if self.training:
            pos_tag_emb = candidate_emb[:, 0, :]  # [B, D]
            x_0, x_t, t, weights = self.diffusion_core.forward_diffusion(seq_emb, pos_tag_emb, mask_seq)
            scores = torch.bmm(candidate_emb, x_0.unsqueeze(2)).squeeze(2)  # [B, N]
            out_dict = {
                'prediction': scores,
                'x_0': x_0,
                'x_t': x_t,
                't': t,
                'weights': weights
            }
        else:
            pos_tag_emb = candidate_emb[:, 0, :]  # [B, D]
            x_T = torch.randn_like(pos_tag_emb)
            x_0 = self.diffusion_core.reverse_diffusion(seq_emb, x_T, mask_seq)
            scores = torch.bmm(candidate_emb, x_0.unsqueeze(2)).squeeze(2)  # [B, N]
            out_dict = {'prediction': scores}

        return out_dict

    def loss(self, out_dict):
        if self.training:
            predictions = out_dict['prediction']  # [B, N]
            pos_tags = out_dict['feed_dict']['item_id'][:, 0]  # [B]
            loss = F.cross_entropy(predictions, torch.zeros(predictions.size(0), dtype=torch.long, device=predictions.device))
            return loss
        return torch.tensor(0.0, device=self.device)


class DiffuRec(SequentialModel, DiffuRecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'hidden_size', 'num_blocks', 'diffusion_steps',
        'lambda_uncertainty', 'noise_schedule', 'emb_dropout',
        'rescale_timesteps', 'num_heads'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = DiffuRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = DiffuRecBase.forward(self, feed_dict)
        out_dict['feed_dict'] = feed_dict
        return out_dict

    def loss(self, out_dict):
        return DiffuRecBase.loss(self, out_dict)