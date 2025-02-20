# -*- coding: UTF-8 -*-
# @Author  : Chao Deng
# @Email   : dengch26@mail2.sysu.edu.cn


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
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--time_size', type=int, default=256,
                            help='Size of embedding vectors.')
        parser.add_argument('--max_time', type=int, default=256,
                            help='Max time intervals.')
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.time_size = args.time_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.max_time = args.max_time
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._base_define_params()
        self.apply(self.init_weights)
        self.R = 0
        self.gram_matrix = 0

        # 在进行训练前获得全部交互时间，并求得最大值、最小值、最小时间间隔，以此正则化时间信息
        time_seqs = []
        # 训练集中给出最小的时间戳
        for u, user_df in corpus.all_df.groupby('user_id'):
            time_seqs.extend(user_df['time'].values.tolist())
        # 测试集中给出最大的时间戳
        for u, user_df in corpus.data_df['test'].groupby('user_id'):
            time_seqs.extend(user_df['time'].values.tolist())
        time_seqs = sorted(set([int(_) for _ in time_seqs]))
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
        #初始化物品Embedding、位置Embedding、时间Embedding
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.t_embeddings = nn.Embedding(self.max_time + 2, self.emb_size)

        # 只使用单层自注意编码区块
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer_Rt(d_model=self.emb_size, d_ff=self.emb_size, d_t=self.time_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        self.check_list = []

        # 算法输入部分
        u_ids = feed_dict['user_id']  # 用户id，RtMIPSRec不对用户特征直接建模，因此不使用用户id信息
        i_ids = feed_dict['item_id']  # 目标item的id，训练时为1阳性item+1阴性item，验证和测试时为1阳性item+999阴性item
        history = feed_dict['history_items']  # 会话的item序列
        t_history = feed_dict['history_times']  # 历史交互的发生时间
        t_target = feed_dict['target_time']  # 目标阳性item的id对应的实际时间，规模与item_id[:, 0]一致，模拟实时推荐效果
        lengths = feed_dict['lengths']  # 用户会话序列的有效长度
        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()

        # 兴趣编码
        interests_sim = self.gram_matrix[history]
        interests_input = interests_sim @ self.i_embeddings.weight
        his_vectors = interests_input

        # 位置编码
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)

        # 时间编码
        # 首先获取每一个交互离最新交互的时间间隔current_interval
        t_history = t_history
        t_history = (t_history - self.min_timestamp).relu()
        t_history = t_history / self.min_interval
        realtime = (t_target - self.min_timestamp) / self.min_interval
        current_interval = realtime.unsqueeze(-1).expand_as(t_history) - t_history
        convert_log_a = torch.pow(torch.tensor(self.max_timestamp_converted), torch.tensor(1. / self.max_time))
        idx = (torch.log(current_interval + 1) / torch.log(convert_log_a)).int()
        time_vectors = self.t_embeddings(idx) # 单调部分完成，但还没有卷积的部分

        # 叠加兴趣编码、位置编码、时间编码得到混合兴趣Embedding
        his_vectors = his_vectors + pos_vectors + time_vectors

        # 进行自注意编码
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int32))
        attn_mask = torch.from_numpy(causality_mask).to(torch.device('cuda'))
        attn_mask_full = torch.ones_like(attn_mask) # RtMIPSRec不同于SASRec架构，其不使用顺序掩码
        for block in self.transformer_block:
            his_vectors, weight_t = block(his_vectors, attn_mask_full) # 分别得到编码后的混合兴趣和频域信息
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # 正则化时间间隔粒度，将其单位调整为秒，然后从周期角度指定频域的边界，这里存在冗余编码
        if self.min_interval == 86400:
            unit_oneday = 1
        elif self.min_interval == 1:
            unit_oneday = 86400
        else:
            unit_oneday = 86400 / self.min_interval
        first_element = unit_oneday * 1
        last_element = unit_oneday * 10 * 365

        # 计算omega
        length = self.time_size
        ratio = (last_element - first_element) / length
        period = (first_element + (ratio * torch.arange(length))).float().to(self.device)
        omega = (2 * torch.pi / period).to(self.device)
        # 通过傅里叶级数公式获取各频段下的相对权重alpha
        time_attenuation = period[None, None, :] / (period[None, None, :] + 0.01 * current_interval[:, :, None]) # 添加周期性损失
        alpha = weight_t * time_attenuation * ((torch.cos(current_interval[:, :, None] * omega[None, None, :]) + 1) / 2)
        numda = alpha.sum(-1)

        # 获取阳性item和阴性item的embedding
        i_vectors = self.i_embeddings(i_ids)
        # 获取加权后的混合兴趣表示
        his_vectors = his_vectors * numda[:, :, None]
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # 输出侧，把所有的vectors放一起求内积，然后求加权匹配值
        prediction = (his_vectors[:, None, :, :] * i_vectors[:, :, None, :])
        prediction = prediction.sum(-1).sum(-1)
        prediction = prediction[:, :] / lengths[:, None]

        return {'prediction': prediction.view(batch_size, -1)}


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

    # 根据交互矩阵获取物品间相似度矩阵
    def get_gram_matrix(self, dataset):
        R = torch.zeros(self.user_num, self.item_num)
        for (user_index, item_index) in zip(dataset.data['user_id'], dataset.data['item_id']):
            R[user_index, item_index] = 1
        self.R = R.to(self.device)

        # 得到一阶相似度矩阵
        row_sum = np.array(R.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isposinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_mat = d_mat.dot(R)
        col_sum = np.array(R.sum(axis=0))
        d_inv = np.power(col_sum, -0.5).flatten()
        d_inv[np.isposinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_mat = norm_mat.dot(d_mat.toarray()).astype(np.float32)
        gram_matrix = norm_mat.T.dot(norm_mat)
        gram_matrix =  torch.Tensor(gram_matrix).to(self.device)

        # 获取取高阶相似度
        item_embedding_r2 = gram_matrix @ self.R.T
        gram_matrix_r2 = item_embedding_r2 @ item_embedding_r2.T
        gram_matrix_r2 =  torch.nn.functional.normalize(gram_matrix_r2)
        gram_matrix_r2 = gram_matrix_r2 / gram_matrix_r2.mean() * gram_matrix.mean()
        gram_matrix = gram_matrix * 0.8 + gram_matrix_r2 * 0.2

        # 取top10%的相似度
        top = int(self.item_num * 0.1)
        indices = torch.topk(gram_matrix, top, dim=1).indices
        gram_matrix_topk = torch.zeros_like(gram_matrix)
        gram_matrix_topk.scatter_(1, indices, gram_matrix.gather(1, indices))
        gram_matrix_topk = torch.nn.functional.normalize(gram_matrix_topk, p=2)
        self.gram_matrix = gram_matrix_topk

        return


    def forward(self, feed_dict):
        out_dict = RtMIPSRecBase.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}
