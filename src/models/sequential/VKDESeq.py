# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" VKDESeq
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

class VKDESeqBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        return parser        

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._base_define_params()
        self.apply(self.init_weights)
        self.R = 0
        self.gram_matrix = 0  # 把item相似矩阵放在base里面


    def _base_define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size] # 每一个用户序列的长度，取值1-20
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        # his_vectors = self.i_embeddings(history)

        interests_sim = self.gram_matrix[history]
        # 3种方式：
        # 1，哈达玛乘一个全部交互，即直接拿相似度矩阵；
        # 2，哈达玛乘一个用户全局交互
        # 3，哈达玛乘一个用户会话内交互
        # user_it = torch.Tensor
        # history_01 = history.scatter_
        # .scatter_(1, indices, gram_matrix.gather(1, indices))
        # interests_sim = interests_sim * 

        # torch.nn.functional.normalize(interests_sim, p=2)


        interests_input = interests_sim @ self.i_embeddings.weight
        his_vectors = interests_input

        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int32)) # 只取下三角的矩阵，表示seq的邻接关系
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask) # transformer的输出维度和输入维度是一样的
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # 只取最后一个item的embedding作为本次训练的预测embedding
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :] # 为什么不直接用冒号？
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        i_vectors = self.i_embeddings(i_ids) # 获取阳性item和阴性item的embedding

        prediction = (his_vectors[:, None, :, :] * i_vectors[:, :, None, :])
        prediction = prediction.sum(-1).sum(-1)
        prediction = prediction[:, :] / lengths[:, None]
        # prediction = (his_vector[:, None, :] * i_vectors).sum(-1) # 获取和阳性item、阴性item的内积，前者越大越好后者越小越好

        u_v = his_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
        i_v = i_vectors

        # 返回一个字典。
        # prediction是预测的内积，训练时返回对两个指定item的内积，测试时返回100个item的id？
        # u_v是预测的embedding
        # i_v是阳性和阴性的embedding
        return {'prediction': prediction.view(batch_size, -1), 'kl': 0, 'u_v': u_v, 'i_v':i_v}


class VKDESeq(SequentialModel, VKDESeqBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = VKDESeqBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def get_gram_matrix(self, dataset):
        R = torch.zeros(self.user_num, self.item_num)
        for (user_index, item_index) in zip(dataset.data['user_id'], dataset.data['item_id']):
            R[user_index, item_index] = 1
        # self.R = R

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

        # 取top500的相似度去做
        indices = torch.topk(gram_matrix, 500, dim=1).indices
        gram_matrix_topk = torch.zeros_like(gram_matrix)
        gram_matrix_topk.scatter_(1, indices, gram_matrix.gather(1, indices))

        gram_matrix_topk = torch.nn.functional.normalize(gram_matrix_topk, p=2)
        self.gram_matrix = gram_matrix_topk

    def forward(self, feed_dict):
        out_dict = VKDESeqBase.forward(self, feed_dict)
        # return {'prediction': out_dict['prediction']}
        return {'prediction': out_dict['prediction'], 'kl': out_dict['kl']}
    
class VKDESeqImpression(ImpressionSeqModel, VKDESeqBase):
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = VKDESeqBase.parse_model_args(parser)
        return ImpressionSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        ImpressionSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return VKDESeqBase.forward(self, feed_dict)