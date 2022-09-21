# Implementation for paper 'Attention on Attention for Image Captioning'
# https://arxiv.org/abs/1908.06954

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

from .AttModel import pack_wrapper, AttModel, Attention
from .TransformerModel import LayerNorm, attention, clones, SublayerConnection, PositionwiseFeedForward


# === 3.2.1 Encoder with AoA ===

# 一个refining module中的multi-head attention模块 (论文figure 4中的浅绿色部分)
class MultiHeadedDotAttention(nn.Module):
    """
    MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=1, scale=opt.multi_head_scale,
                            do_aoa=opt.refine_aoa, norm_q=0, dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))
    """
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1,
                 use_output_layer=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # h: attention heads的数量, 8
        # d_model：rnn的规模 (每一层的#hidden nodes), 1024
        # scale: 是否对q,k,v做scaling（论文的式(10)）, 1

        # 认为d_v = d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        # 训练的时候就设为1
        self.project_k_v = project_k_v

        # 是否对query进行标准化
        # 训练的时候设为0，不normalize
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x  # 就是原封不动
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)  # 1 or 3 层

        # 是否在multi-head attention后输出linear layer
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # 是否加入aoa模块
        # 训练的时候就是设为1，即使用AoA
        self.use_aoa = do_aoa
        if self.use_aoa:
            # 1: nn.Linear((1 + scale) * d_model, 2 * d_model) -> (2048, 2048)
            # 2: nn.GLU()
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # 对aoa层的输入进行dropout操作
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        # 根据文章3.2.1的最后一段，如果在Encoder中使用了aoa，就把原始的feed-forward layer drop掉
        if self.use_aoa or not use_output_layer:
            # 去掉output linear layer
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # 对所有heads使用mask
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k 
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        # 对batch中的所有projected vectors使用attention机制
        # attention: 计算'Scaled Dot Product Attention'
        # p_attn: F.softmax
        # x是matmul(p_attn, value)，见论文式(10). x就是要Concat的所有head_i的矩阵
        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)

        # 用view进行"Concat"
        # x即为输入AoA的V_hat，论文里的式(8)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        if self.use_aoa:
            # Apply AoA
            # torch.cat([x, query]: AoA的输入（两个，V_hat和原始的Q，见figure 4）
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x


# 一整个refining module (论文的figure 4)
class AoA_Refiner_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        size: opt.rnn_size
        self_attn: MultiHeadedDotAttention
        feed_forward: None, 前馈
        dropout: 0.1
        """
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        # 进行residual connection
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x


# 含有6个refining module(layer)的整个refining块儿 (论文figure 3的蓝色部分)
class AoA_Refiner_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Refiner_Core, self).__init__()
        # 定义AoA中使用的attn
        attn = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=1, scale=opt.multi_head_scale,
                                       do_aoa=opt.refine_aoa, norm_q=0, dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))
        # refining module
        # 每一层AoA中的attention部分用的都是MultiHeadedDotAttention
        layer = AoA_Refiner_Layer(opt.rnn_size, attn,
                                  PositionwiseFeedForward(opt.rnn_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        # 建立6层一样的refining module
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# === 3.2.2 Decoder with AoA ===
class AoA_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.decoder_type = getattr(opt, 'decoder_type', 'AoA')
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        if self.decoder_type == 'AoA':
            # aoa层
            self.att2ctx = nn.Sequential(
                nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'LSTM':
            # LSTM层
            self.att2ctx = nn.LSTMCell(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size)
        else:
            # 线性层
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size),
                                         nn.ReLU())

        if opt.use_multi_head == 2:
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0,
                                                     scale=opt.multi_head_scale, use_output_layer=0, do_aoa=0, norm_q=1)
        else:
            self.attention = Attention(opt)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        # state[0][1] is the context vector at the last step
        # 构建LSTM层（论文figure 5的下小半部分）
        # 论文(13). c_att表示的是m_t
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1),
                                     (state[0][0], state[1][0]))

        # 将LSTM的输出和A输入attention部分（论文figure 5的浅绿色部分）
        if self.use_multi_head == 2:
            att = self.attention(h_att, p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model),
                                 p_att_feats.narrow(2, self.multi_head_scale * self.d_model,
                                                    self.multi_head_scale * self.d_model), att_masks)
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        # 把LSTM的输出h和attention的输出a合并，输入aoa
        # figure 5的蓝灰色部分，式(14)
        ctx_input = torch.cat([att, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
        else:
            output = self.att2ctx(ctx_input)
            # 存储context vector，放到state[0][1]
            state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))

        if self.out_res:  # <- 0
            # 残差连接
            output = output + h_att

        output = self.out_drop(output)
        return output, state


class AoAModel(AttModel):
    def __init__(self, opt):
        super(AoAModel, self).__init__(opt)
        self.num_layers = 2
        # 做mean pooling
        self.use_mean_feats = getattr(opt, 'mean_feats', 1)
        if opt.use_multi_head == 2:
            del self.ctx2att
            self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.multi_head_scale * opt.rnn_size)

        if self.use_mean_feats:
            del self.fc_embed

        # refiner -> encoder
        if opt.refine:
            self.refiner = AoA_Refiner_Core(opt)
        else:
            self.refiner = lambda x, y: x

        # core -> decoder
        self.core = AoA_Decoder_Core(opt)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.refiner(att_feats, att_masks)

        if self.use_mean_feats:
            # meaning pooling
            if att_masks is None:
                mean_feats = torch.mean(att_feats, dim=1)
            else:
                mean_feats = (torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1))
        else:
            mean_feats = self.fc_embed(fc_feats)

        # Project the attention feats
        # 是为了减少内存占用和计算量
        p_att_feats = self.ctx2att(att_feats)

        return mean_feats, att_feats, p_att_feats, att_masks
