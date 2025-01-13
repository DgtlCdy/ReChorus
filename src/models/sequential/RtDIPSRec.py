# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" RtMIPSRec
Reference:
    "Self-attentive Sequential Recommendation"
    Kang et al., IEEE'2018.
Note:
    When incorporating position embedding, we make the position index start from the most recent interaction.
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import utils

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel
from utils import layers

class RtMIPSRecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--time_size', type=int, default=256,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--time_max', type=int, default=256,
                            help='Max time intervals.')
        return parser        

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.time_size = args.time_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.max_time = args.time_max
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._base_define_params()
        self.apply(self.init_weights)
        self.R = 0
        self.gram_matrix = 0  # 把item相似矩阵放在base里面

        # 获得全部时间，并求得最大值、最小值、最小时间间隔，然后根据这些参数建模时间间隔embedding、确定索引方式
        time_seqs = []
        # 训练集中给出最小的时间戳
        for u, user_df in corpus.all_df.groupby('user_id'):
            time_seqs.extend(user_df['time'].values.tolist())
        # 测试集中给出最大的时间戳
        for u, user_df in corpus.data_df['test'].groupby('user_id'):
            time_seqs.extend(user_df['time'].values.tolist())
        time_seqs = sorted(set([int(_) for _ in time_seqs]))
        # time_b = torch.Tensor(time_seqs + [0xFFFFFFFF]).int()
        # time_a = torch.Tensor([0] + time_seqs).int()
        # time_intervals = time_b - time_a
        # self.min_interval = min(time_intervals).item()
        self.min_timestamp = time_seqs[0]
        self.max_timestamp = time_seqs[-1]
        self.min_interval = 0xFFFFFFFF
        for idx in range(len(time_seqs) - 1):
            self.min_interval = min(self.min_interval, time_seqs[idx+1] - time_seqs[idx])
        if self.min_interval == 0:
            self.min_interval = 1
        self.min_timestamp_converted = 0
        self.max_timestamp_converted = (self.max_timestamp - self.min_timestamp) / self.min_interval


    def _base_define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.t_embeddings_gm = nn.Embedding(self.max_time + 2, self.emb_size)
        self.t_embeddings_gp_k = nn.Embedding(self.max_time + 2, self.emb_size)
        self.t_embeddings_gp_v = nn.Embedding(self.max_time + 2, self.emb_size)
        # self.t_embeddings_sa = nn.Embedding(self.max_time + 2, self.emb_size)
        # self.t_embeddings_ffn = nn.Embedding(self.max_time + 2, self.emb_size)

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer_Ti(d_model=self.emb_size, d_ff=self.emb_size, d_t=self.time_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        t_history = feed_dict['history_times']  # [batch_size, history_max]

        # 这里作为实时推荐任务，应当为feed_dict添加item_id对应的time，包括训练、验证、测试的三个时间
        t_target = feed_dict['target_time']  # 应当与i_ids一致

        lengths = feed_dict['lengths']  # [batch_size] # 每一个用户序列的长度，取值1-20

        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()

        interests_sim = self.gram_matrix[history]
        # 4种构建基于相似的兴趣的方式：
        # 0，不使用交互，传入自身Embedding直接作为兴趣
        # his_vectors = self.i_embeddings(history)
        # 1，直接拿相似度矩阵，哈达玛乘一个全1向量
        interests_sim = interests_sim
        interests_input = interests_sim @ self.i_embeddings.weight
        his_vectors = interests_input
        # 2，哈达玛乘一个用户全局交互
        # user_interaction = self.R[u_ids]
        # interests_sim = interests_sim[:, :, :] * user_interaction[:, None, :]
        # interests_sim = torch.nn.functional.normalize(interests_sim, p=2)
        # interests_input = interests_sim @ self.i_embeddings.weight
        # his_vectors = interests_input
        # 3，哈达玛乘一个用户会话内交互，即lengths个交互
        # user_interaction = torch.zeros(batch_size, self.item_num).to(self.device)
        # for idx in range(batch_size):
        #     user_interaction[idx, history[idx, :lengths[idx]]] = 1
        # interests_sim = interests_sim[:, :, :] * user_interaction[:, None, :]
        # interests_sim = torch.nn.functional.normalize(interests_sim, p=2)
        # interests_input = interests_sim @ self.i_embeddings.weight
        # his_vectors = interests_input

        # 获取每一个交互离最新交互的时间间隔current_interval
        t_history = t_history
        t_history = (t_history - self.min_timestamp).relu()
        t_history = t_history / self.min_interval
        max_values, _ = torch.max(t_history, dim=1)
        realtime = (t_target - self.min_timestamp) / self.min_interval
        # current_interval = max_values.unsqueeze(-1).expand_as(t_history) - t_history
        current_interval = realtime.unsqueeze(-1).expand_as(t_history) - t_history

        # 将时间间隔转化为时间Embedding
        # # 方案1：小于1的幂次
        # convert_pow = torch.log(torch.tensor(self.max_time)) / torch.log(torch.tensor(self.max_timestamp_converted))
        # idx = torch.pow(current_interval, convert_pow).int()
        # # 方案2：线性
        # convert_line = torch.tensor(self.max_time) / torch.tensor(self.max_timestamp_converted)
        # idx = (current_interval * convert_line).int()
        # 方案3：对数
        convert_log_a = torch.pow(torch.tensor(self.max_timestamp_converted), torch.tensor(1. / self.max_time))
        # convert_log = torch.exp(convert_log_a), convert_log_a = torch.log(convert_log)
        idx = (torch.log(current_interval + 1) / torch.log(convert_log_a)).int()

        # 获取单调和周期性的时间Embedding
        t_ebds_m = self.t_embeddings_gm(idx) # 单调部分完成，但还没有卷积的部分

        # idx_g = torch.Tensor(range(self.max_time)).int().to(torch.device('cuda'))
        # scores_g = his_vectors @ self.t_embeddings_gp_k(idx_g).T / (self.emb_size ** 0.5)
        # scores_valid = torch.zeros_like(scores_g)
        # for i in range(his_vectors.size(0)):
        #     for j in range(lengths[i]):
        #         scores_valid[i, j, :idx[i, j]] = 1
        # 这个方法出现未知的cuda问题，暂时不用
        # valid_indice = torch.tril(torch.ones(128, 128), diagonal=0).int().to(self.device)
        # scores_valid = valid_indice[idx]

        # scores_g_weighted = torch.softmax(scores_g, dim=-1) * scores_valid
        # scores_g_weighted = torch.nn.functional.normalize(scores_g_weighted, p=1, dim=-1)
        # t_ebds_p = scores_g_weighted @ self.t_embeddings_gp_v(idx_g)

        # scores_g_weighted = torch.zeros_like(scores_g)
        # for i in range(his_vectors.size(0)):
        #     for j in range(lengths[i]):
        #         # i_vector = his_vectors[i, j, :]
        #         i_idx = idx[i, j]
        #         idx_v = torch.Tensor(range(i_idx)).int().to(torch.device('cuda'))
        #         # 基于注意力，将得到的时间Embedding传给t_ebds_p
        #         scores = scores_g[i, j, :i_idx]
        #         scores_g_weighted[i, j, :i_idx] = torch.softmax(scores, dim=-1)
        #         # scores = torch.softmax(scores, dim=-1)
        #         # t_ebds_p[i, j, :]  = scores @ self.t_embeddings_gp_v(idx_v)
        # t_ebds_p = scores_g_weighted @ self.t_embeddings_gp_v(idx_g)

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)

        # his_vectors = his_vectors + pos_vectors
        # his_vectors = his_vectors + pos_vectors + t_ebds_sa
        his_vectors = his_vectors + pos_vectors + t_ebds_m

        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int32)) # 只取下三角的矩阵，表示seq的邻接关系
        attn_mask = torch.from_numpy(causality_mask).to(torch.device('cuda'))
        attn_mask_full = torch.ones_like(attn_mask)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors, weight_t = block(his_vectors, attn_mask_full) # transformer的输出维度和输入维度是一样的
            # his_vectors = block(his_vectors, attn_mask) # transformer的输出维度和输入维度是一样的
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # 针对获取的时间取旋转的权重
        # 维度：256*20*time_size
        # 其实只需要知道最短时间间隔，也就是时间单位即可，amazon的数据集，最小时间间隔是86400秒，可以先用直接的时间来估计
        # 现在已知current_interval，即每个兴趣的相对时间间隔，而且是调整好的，现在要求出对应的每个权重
        # 假设周期最长为10年
        # 定义第一个元素、最后一个元素和长度
        # first_element = 3600 * 24
        # last_element = 3600 * 24 * 365
        if self.min_interval == 86400:
            unit_oneday = 1
        elif self.min_interval == 1:
            unit_oneday = 86400
        else:
            unit_oneday = 86400 / self.min_interval
        first_element = unit_oneday * 1
        last_element = unit_oneday * 10 * 365
        length = self.time_size
        # 计算公比
        # ratio = (last_element / first_element) ** (1 / (length - 1))
        # period = first_element * (ratio ** torch.arange(length)).float()
        # omega = (2 * torch.pi / period).to(self.device)
        # omega = (period / (2 * torch.pi) / last_element).to(self.device)
        # 计算公差
        ratio = (last_element - first_element) / length
        period = (first_element + (ratio * torch.arange(length))).float().to(self.device)
        omega = (2 * torch.pi / period).to(self.device)
        # omega = (period / (2 * torch.pi) / last_element).to(self.device)

        time_attenuation = period[None, None, :] / (period[None, None, :] + 0.01 * current_interval[:, :, None])


        # weight_t_added = weight_t * ((torch.cos(t_history[:, :, None] * omega[None, None, :]) + 1) / 2)
        # weight_t_added = weight_t * ((torch.cos(current_interval[:, :, None] * omega[None, None, :]) + 1) / 2)
        weight_t_added = weight_t * time_attenuation * ((torch.cos(current_interval[:, :, None] * omega[None, None, :]) + 1) / 2)

        weight_t_added = weight_t_added.sum(-1)
        his_vectors = his_vectors * weight_t_added[:, :, None]
        his_vectors = his_vectors * valid_his[:, :, None].float()


        # 只取最后一个item的embedding作为本次训练的预测embedding
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :] # 为什么不直接用冒号？
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        i_vectors = self.i_embeddings(i_ids) # 获取阳性item和阴性item的embedding

        # 输出侧
        # 方法0：对最后一个输出embedding求内积
        # prediction = (his_vector[:, None, :] * i_vectors).sum(-1) # 获取和阳性item、阴性item的内积，前者越大越好后者越小越好
        # 方法1：把所有的vectors放一起求内积均值
        prediction = (his_vectors[:, None, :, :] * i_vectors[:, :, None, :])
        prediction = prediction.sum(-1).sum(-1)
        prediction = prediction[:, :] / lengths[:, None]


        u_v = his_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
        i_v = i_vectors

        # 返回一个字典。
        # prediction是预测的内积，训练时返回对两个指定item的内积，测试时返回100个item的id？
        # u_v是预测的embedding
        # i_v是阳性和阴性的embedding
        return {'prediction': prediction.view(batch_size, -1), 'kl': 0, 'u_v': u_v, 'i_v':i_v}


class RtMIPSRec(SequentialModel, RtMIPSRecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = RtMIPSRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def get_gram_matrix(self, dataset):
        R = torch.zeros(self.user_num, self.item_num)
        for (user_index, item_index) in zip(dataset.data['user_id'], dataset.data['item_id']):
            R[user_index, item_index] = 1
        self.R = R.to(self.device)

        row_sum = np.array(R.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten() #根号度分之一
        d_inv[np.isposinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv) # 对角的度矩阵
        norm_mat = d_mat.dot(R)
        col_sum = np.array(R.sum(axis=0))
        d_inv = np.power(col_sum, -0.5).flatten()
        d_inv[np.isposinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_mat = norm_mat.dot(d_mat.toarray()).astype(np.float32)
        gram_matrix = norm_mat.T.dot(norm_mat)
        gram_matrix =  torch.Tensor(gram_matrix).to(self.device)

        # 方法0：取原生的相似度
        # self.gram_matrix = gram_matrix
        # return

        # 方法1：取高阶相似度
        # item_embedding_r2 = gram_matrix @ self.R.T
        # gram_matrix_r2 = item_embedding_r2 @ item_embedding_r2.T
        # gram_matrix_r2 =  torch.nn.functional.normalize(gram_matrix_r2)
        # gram_matrix_r2 = gram_matrix_r2 / gram_matrix_r2.mean() * gram_matrix.mean()
        # gram_matrix = gram_matrix * 0.5 + gram_matrix_r2 * 0.5
        # self.gram_matrix = gram_matrix * 0.7 + gram_matrix_r2 * 0.3
        # return

        # 方法2：取top相似度
        # 取top500的相似度去做
        indices = torch.topk(gram_matrix, 500, dim=1).indices
        gram_matrix_topk = torch.zeros_like(gram_matrix)
        gram_matrix_topk.scatter_(1, indices, gram_matrix.gather(1, indices))

        gram_matrix_topk = torch.nn.functional.normalize(gram_matrix_topk, p=2)
        self.gram_matrix = gram_matrix_topk


    def forward(self, feed_dict):
        out_dict = RtMIPSRecBase.forward(self, feed_dict)
        # return {'prediction': out_dict['prediction']}
        return {'prediction': out_dict['prediction'], 'kl': out_dict['kl']}
    
class RtMIPSRecImpression(ImpressionSeqModel, RtMIPSRecBase):
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = RtMIPSRecBase.parse_model_args(parser)
        return ImpressionSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        ImpressionSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return RtMIPSRecBase.forward(self, feed_dict)