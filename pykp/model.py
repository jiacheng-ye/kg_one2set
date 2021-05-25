import torch.nn as nn

import pykp.utils.io as io
from pykp.decoder.transformer import TransformerSeq2SeqDecoder
from pykp.encoder.transformer import TransformerSeq2SeqEncoder
from pykp.modules.position_embed import get_sinusoid_encoding_table


class Seq2SeqModel(nn.Module):
    """Container module with an encoder, decoder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()
        embed = nn.Embedding(opt.vocab_size, opt.word_vec_size, opt.vocab["word2idx"][io.PAD_WORD])
        self.init_emb(embed)
        pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(3000, opt.word_vec_size, padding_idx=opt.vocab["word2idx"][io.PAD_WORD]),
            freeze=True)
        self.encoder = TransformerSeq2SeqEncoder.from_opt(opt, embed, pos_embed)
        self.decoder = TransformerSeq2SeqDecoder.from_opt(opt, embed, pos_embed)

    def init_emb(self, embed):
        """Initialize weights."""
        initrange = 0.1
        embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_lens, input_tgt, src_oov, max_num_oov, src_mask):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :return:
        """
        # Encoding
        memory_bank = self.encoder(src, src_lens, src_mask)
        state = self.decoder.init_state(memory_bank, src_mask)
        decoder_dist_all, attention_dist_all = self.decoder(input_tgt, state, src_oov, max_num_oov)
        return decoder_dist_all, attention_dist_all


