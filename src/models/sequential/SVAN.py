# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" SVAN
Reference:
    "Self-attentive Sequential Recommendation"
    Kang et al., IEEE'2018.
Note:
    When incorporating position embedding, we make the position index start from the most recent interaction.
"""

import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel
from utils import layers

class SVANBase(object):
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
        self.vae_encoder = nn.Sequential()
        self.vae_decoder = nn.Sequential()
        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

        enc_dims = [1000, 64]
        dec_dims = [64, 1000]
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            # 在倒数第二层分裂开，一半均值，一半方差
            if i == len(enc_dims) - 2:
                out_dim = out_dim * 2
            self.vae_encoder.add_module(name='Encoder_Linear_%s'%i, module=nn.Linear(in_dim, out_dim))
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            # 解码器不需要分裂开，通过重采样来传
            self.vae_decoder.add_module(name='Decoder_Linear_%s'%i, module=nn.Linear(in_dim, out_dim))

        self.decode_transformer_block = nn.ModuleList([
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
        his_vectors = self.i_embeddings(history)

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

        # # 进入VAE
        # x = self.vae_encoder(his_vectors.flatten())
        # mean, logvar = x[:, :(len(x[0] - 1)//2)], x[:, (len(x[0] - 1)//2):-1]
        # stddev = torch.exp(0.5 * logvar)
        # epsilon = torch.randn_like(stddev)
        # if self.training:
        #     z = mean + epsilon * stddev
        # else:
        #     self.optim.zero_grad()
        #     z = mean
        # his_vectors = self.vae_decoder(z)

        for block in self.decode_transformer_block:
            his_vectors = block(his_vectors, attn_mask) # transformer的输出维度和输入维度是一样的

        his_vectors = his_vectors * valid_his[:, :, None].float()

        # 只取最后一个item的embedding作为本次训练的预测embedding
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :] # 为什么不直接用冒号？
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        i_vectors = self.i_embeddings(i_ids) # 获取阳性item和阴性item的embedding
        prediction = (his_vector[:, None, :] * i_vectors).sum(-1) # 获取和阳性item、阴性item的内积，前者越大越好后者越小越好

        u_v = his_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
        i_v = i_vectors

        # 返回一个字典。
        # prediction是预测的内积，训练时返回对两个指定item的内积，测试时返回100个item的id？
        # u_v是预测的embedding
        # i_v是阳性和阴性的embedding
        return {'prediction': prediction.view(batch_size, -1), 'u_v': u_v, 'i_v':i_v}


class SVAN(SequentialModel, SVANBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = SVANBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = SVANBase.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}
    
class SVANImpression(ImpressionSeqModel, SVANBase):
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = SVANBase.parse_model_args(parser)
        return ImpressionSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        ImpressionSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return SVANBase.forward(self, feed_dict)