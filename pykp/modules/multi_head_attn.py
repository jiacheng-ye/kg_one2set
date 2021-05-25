import torch
import torch.nn as nn
from pykp.utils.seq2seq_state import TransformerState
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Attention is all you need

    """
    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.0, layer_idx: int = None,
                 fix_kp_num_len=False, max_kp_num=20):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = d_model // n_head
        self.layer_idx = layer_idx
        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.fix_kp_num_len = fix_kp_num_len
        self.max_kp_num = max_kp_num
        self.reset_parameters()

    def forward(self, query, key, value, key_mask=None, attn_mask=None, state=None):
        """

        :param query: batch x seq x dim
        :param key: batch x seq x dim
        :param value: batch x seq x dim
        :param key_mask: batch x seq 用于指示哪些key不要attend到；注意到mask为1的地方是要attend到的
        :param attn_mask: seq x seq, 用于mask掉attention map。 主要是用在训练时decoder端的self attention，下三角为1
        :param state: 过去的信息，在inference的时候会用到，比如encoder output、decoder的prev kv。这样可以减少计算。
        :return:
        """
        assert key.size() == value.size()
        if state is not None:
            assert self.layer_idx is not None
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()

        q = self.q_proj(query)  # batch x seq x dim
        q *= self.scaling
        k = v = None
        prev_k = prev_v = None

        # 从state中取kv
        if isinstance(state, TransformerState):  # 说明此时在inference阶段
            if qkv_same:  # 此时在decoder self attention
                prev_k = state.decoder_prev_key[self.layer_idx]
                prev_v = state.decoder_prev_value[self.layer_idx]
            else:  # 此时在decoder-encoder attention，直接将保存下来的key装载起来即可
                k = state.encoder_key[self.layer_idx]
                v = state.encoder_value[self.layer_idx]

        if k is None:
            k = self.k_proj(key)
            v = self.v_proj(value)

        if prev_k is not None:
            if self.fix_kp_num_len:
                batch_size, max_len, d = prev_k.size()
                prev_k = prev_k.reshape(batch_size, self.max_kp_num, -1, d)
                prev_v = prev_v.reshape(batch_size, self.max_kp_num, -1, d)
                k = torch.cat((prev_k, k.unsqueeze(-2)), dim=-2).reshape(batch_size, -1, d)
                v = torch.cat((prev_v, v.unsqueeze(-2)), dim=-2).reshape(batch_size, -1, d)
            else:
                k = torch.cat((prev_k, k), dim=1)
                v = torch.cat((prev_v, v), dim=1)

        # 更新state
        if isinstance(state, TransformerState):
            if qkv_same:
                state.decoder_prev_key[self.layer_idx] = k
                state.decoder_prev_value[self.layer_idx] = v
            else:
                state.encoder_key[self.layer_idx] = k
                state.encoder_value[self.layer_idx] = v

        # 开始计算attention
        batch_size, q_len, d_model = query.size()
        k_len, v_len = k.size(1), v.size(1)
        q = q.reshape(batch_size, q_len, self.n_head, self.head_dim)
        k = k.reshape(batch_size, k_len, self.n_head, self.head_dim)
        v = v.reshape(batch_size, v_len, self.n_head, self.head_dim)

        attn_weights = torch.einsum('bqnh,bknh->bqkn', q, k)  # bs,q_len,k_len,n_head

        if key_mask is not None:
            _key_mask = ~key_mask[:, None, :, None].bool()  # batch,1,k_len,1
            attn_weights = attn_weights.masked_fill(_key_mask, -float('inf'))

        if attn_mask is not None:
            _attn_mask = attn_mask[None, :, :, None].eq(0)  # 1,q_len,k_len,n_head
            attn_weights = attn_weights.masked_fill(_attn_mask, -float('inf'))

        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.einsum('bqkn,bknh->bqnh', attn_weights, v)  # batch,q_len,n_head,head_dim
        output = output.reshape(batch_size, q_len, -1)
        output = self.out_proj(output)  # batch,q_len,dim

        return output, attn_weights

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def set_layer_idx(self, layer_idx):
        self.layer_idx = layer_idx
