"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from pykp.modules.multi_head_attn import MultiHeadAttention
from pykp.utils.seq2seq_state import TransformerState
import torch.nn.functional as F
import math


class TransformerSeq2SeqDecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, dim_ff=2048, dropout=0.1, layer_idx=None,
                 fix_kp_num_len=False, max_kp_num=20):
        """
        :param int d_model: 输入、输出的维度
        :param int n_head: 多少个head，需要能被d_model整除
        :param int dim_ff:
        :param float dropout:
        :param int layer_idx: layer的编号
        """
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_idx = layer_idx  # 记录layer的层索引，以方便获取state的信息

        self.self_attn = MultiHeadAttention(d_model, n_head, dropout, layer_idx,
                                            fix_kp_num_len, max_kp_num)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.encoder_attn = MultiHeadAttention(d_model, n_head, dropout, layer_idx)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x, encoder_output, encoder_mask=None, self_attn_mask=None, state=None):
        """
        :param x: (batch, seq_len, dim), decoder端的输入
        :param encoder_output: (batch,src_seq_len,dim), encoder的输出
        :param encoder_mask: batch,src_seq_len, 为1的地方需要attend
        :param self_attn_mask: seq_len, seq_len，下三角的mask矩阵，只在训练时传入
        :param TransformerState state: 只在inference阶段传入
        :return:
        """

        # self attention part
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              attn_mask=self_attn_mask,
                              state=state)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # encoder attention part
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, attn_weight = self.encoder_attn(query=x,
                                           key=encoder_output,
                                           value=encoder_output,
                                           key_mask=encoder_mask,
                                           state=state)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weight


class TransformerSeq2SeqDecoder(nn.Module):
    def __init__(self, embed, pos_embed,
                 d_model=512, num_layers=6, n_head=8, dim_ff=2048, dropout=0.1, copy_attn=False,
                 fix_kp_num_len=False, max_kp_len=6, max_kp_num=20):
        """
        :param embed: 输入token的embedding
        :param nn.Module pos_embed: 位置embedding
        :param int d_model: 输出、输出的大小
        :param int num_layers: 多少层
        :param int n_head: 多少个head
        :param int dim_ff: FFN 的中间大小
        :param float dropout: Self-Attention和FFN中的dropout的大小
        """
        super().__init__()

        self.embed = embed
        self.pos_embed = pos_embed

        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_fc = nn.Linear(self.embed.embedding_dim, d_model)
        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqDecoderLayer(d_model, n_head, dim_ff, dropout, layer_idx,
                                                                          fix_kp_num_len, max_kp_num)
                                           for layer_idx in range(num_layers)])
        self.embed_scale = math.sqrt(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.vocab_size = self.embed.num_embeddings
        self.output_fc = nn.Linear(self.d_model, self.embed.embedding_dim)
        self.output_layer = nn.Linear(self.embed.embedding_dim, self.vocab_size, bias=False)

        self.copy_attn = copy_attn
        if copy_attn:
            self.p_gen_linear = nn.Linear(self.embed.embedding_dim, 1)

        self.fix_kp_num_len = fix_kp_num_len
        if self.fix_kp_num_len:
            self.max_kp_len = max_kp_len
            self.max_kp_num = max_kp_num
            self.control_code = nn.Embedding(max_kp_num, self.embed.embedding_dim)
            self.control_code.weight.data.uniform_(-0.1, 0.1)
            self.self_attn_mask = self._get_self_attn_mask(max_kp_num, max_kp_len)

    @classmethod
    def from_opt(cls, opt, embed, pos_embed):
        return cls(embed,
                   pos_embed,
                   num_layers=opt.dec_layers,
                   d_model=opt.d_model,
                   n_head=opt.n_head,
                   dim_ff=opt.dim_ff,
                   dropout=opt.dropout,
                   copy_attn=opt.copy_attention,
                   fix_kp_num_len=opt.fix_kp_num_len,
                   max_kp_len=opt.max_kp_len,
                   max_kp_num=opt.max_kp_num)

    def forward_seg(self, state):
        encoder_output = state.encoder_output
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        control_idx = torch.arange(0, self.max_kp_num).long().to(device).reshape(1, -1).repeat(batch_size, 1)
        control_embed = self.control_code(control_idx)

        return control_embed

    def forward(self, tokens, state, src_oov, max_num_oov, control_embed=None):
        """
        :param torch.LongTensor tokens: batch x tgt_len，decode的词
        :param TransformerState state: 用于记录encoder的输出以及decode状态的对象，可以通过init_state()获取
        :return: bsz x max_len x vocab_size; 如果return_attention=True, 还会返回bsz x max_len x encode_length
        """

        encoder_output = state.encoder_output
        encoder_mask = state.encoder_mask
        device = tokens.device

        if self.fix_kp_num_len:
            decode_length = state.decode_length // self.max_kp_num
            assert decode_length < tokens.size(2), "The decoded tokens in State should be less than tokens."
            tokens = tokens[:, :, decode_length:]
            batch_size, max_kp_num, kp_len = tokens.size()
            max_tgt_len = max_kp_num * kp_len

            position = torch.arange(decode_length, decode_length + kp_len).long().to(device).reshape(1, 1, -1)
            position_embed = self.pos_embed(position)

            word_embed = self.embed_scale * self.embed(tokens)
            embed = self.input_fc(word_embed) + position_embed + control_embed.reshape(batch_size, max_kp_num, 1, -1)
            x = F.dropout(embed, p=self.dropout, training=self.training)
            x = x.reshape(batch_size, max_kp_num * kp_len, -1)

            if self.self_attn_mask.device is not tokens.device:
                self.self_attn_mask = self.self_attn_mask.to(tokens.device)

            if kp_len > 1:  # training
                self_attn_mask = self.self_attn_mask
            else:
                self_attn_mask = self.self_attn_mask.reshape(max_kp_num, self.max_kp_len, max_kp_num, self.max_kp_len)\
                    [:, decode_length, :, :decode_length + 1] \
                    .reshape(max_kp_num, max_kp_num * (decode_length + 1))

            for layer in self.layer_stacks:
                x, attn_dist = layer(x=x,
                                     encoder_output=encoder_output,
                                     encoder_mask=encoder_mask,
                                     self_attn_mask=self_attn_mask,
                                     state=state
                                     )
        else:
            assert state.decode_length < tokens.size(1), "The decoded tokens in State should be less than tokens."
            tokens = tokens[:, state.decode_length:]

            position = torch.arange(state.decode_length, state.decode_length + tokens.size(1)).long().to(device)[
                None]
            position_embed = self.pos_embed(position)

            batch_size, max_tgt_len = tokens.size()
            word_embed = self.embed_scale * self.embed(tokens)
            embed = self.input_fc(word_embed) + position_embed
            x = F.dropout(embed, p=self.dropout, training=self.training)
            if max_tgt_len > 1:
                self_attn_mask = self._get_triangle_mask(tokens)
            else:
                self_attn_mask = None

            for layer in self.layer_stacks:
                x, attn_dist = layer(x=x,
                                     encoder_output=encoder_output,
                                     encoder_mask=encoder_mask,
                                     self_attn_mask=self_attn_mask,
                                     state=state
                                     )
        x = self.layer_norm(x)  # batch, tgt_len, dim
        x = self.output_fc(x)

        vocab_dist = F.softmax(self.output_layer(x), -1)
        attn_dist = attn_dist[:, :, :, 0]

        if self.copy_attn:
            p_gen = self.p_gen_linear(x).sigmoid()

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if max_num_oov > 0:
                extra_zeros = vocab_dist_.new_zeros((batch_size, max_tgt_len, max_num_oov))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=-1)

            final_dist = vocab_dist_.scatter_add(2, src_oov.unsqueeze(1).expand_as(attn_dist_), attn_dist_)
            assert final_dist.size() == torch.Size([batch_size, max_tgt_len, self.vocab_size + max_num_oov])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, max_tgt_len, self.vocab_size])
        return final_dist, attn_dist

    def init_state(self, encoder_output, encoder_mask):
        """
        初始化一个TransformerState用于forward
        :param torch.FloatTensor encoder_output: bsz x max_len x d_model, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x max_len, 为1的位置需要attend。
        :return: TransformerState
        """
        if isinstance(encoder_output, torch.Tensor):
            encoder_output = encoder_output
        elif isinstance(encoder_output, (list, tuple)):
            encoder_output = encoder_output[0]  # 防止是LSTMEncoder的输出结果
        else:
            raise TypeError("Unsupported `encoder_output` for TransformerSeq2SeqDecoder")
        state = TransformerState(encoder_output, encoder_mask, num_decoder_layer=self.num_layers)
        return state

    @staticmethod
    def _get_triangle_mask(tokens):
        tensor = tokens.new_ones(tokens.size(1), tokens.size(1))
        return torch.tril(tensor).byte()

    @staticmethod
    def _get_self_attn_mask(max_kp_num, max_kp_len):
        mask = torch.ones(max_kp_num * max_kp_len, max_kp_num * max_kp_len)
        mask = torch.tril(mask).bool()
        for i in range(1, max_kp_num + 1):
            mask[i * max_kp_len:(i + 1) * max_kp_len, :i * max_kp_len] = 0
        return mask
