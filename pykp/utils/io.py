# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.utils.data

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
SEP_WORD = '<sep>'
DIGIT = '<digit>'
PEOS_WORD = '<peos>'
NULL_WORD = '<null>'


class KeyphraseDataset(torch.utils.data.Dataset):
    def __init__(self, examples, word2idx, idx2word, device, load_train=True,
                 fix_kp_num_len=False, max_kp_len=6, max_kp_num=20, seperate_pre_ab=False):
        keys = ['src', 'src_oov', 'oov_dict', 'oov_list', 'src_str', 'trg_str', 'trg', 'trg_copy']

        filtered_examples = []

        for e in examples:
            filtered_example = {}
            for k in keys:
                filtered_example[k] = e[k]
            if 'oov_list' in filtered_example:
                filtered_example['oov_number'] = len(filtered_example['oov_list'])
            filtered_examples.append(filtered_example)

        self.examples = filtered_examples
        self.word2idx = word2idx
        self.id2xword = idx2word
        self.load_train = load_train
        self.device = device

        # for one2set
        self.fix_kp_num_len = fix_kp_num_len
        if self.fix_kp_num_len:
            self.max_kp_len = max_kp_len
            self.max_kp_num = max_kp_num
            self.seperate_pre_ab = seperate_pre_ab

    @classmethod
    def build(cls, examples, opt, load_train):
        return cls(examples,
                   device=opt.device,
                   word2idx=opt.vocab['word2idx'],
                   idx2word=opt.vocab['idx2word'],
                   load_train=load_train,
                   fix_kp_num_len=opt.fix_kp_num_len,
                   max_kp_len=opt.max_kp_len,
                   max_kp_num=opt.max_kp_num,
                   seperate_pre_ab=opt.seperate_pre_ab)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, input_list):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = self.word2idx[PAD_WORD] * np.ones((len(input_list), max_seq_len))

        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]

        padded_batch = torch.LongTensor(padded_batch)

        input_mask = torch.ne(padded_batch, self.word2idx[PAD_WORD]).type(torch.FloatTensor)

        return padded_batch, input_list_lens, input_mask

    def _pad2d(self, input_list):
        input_list_lens = [[len(t) for t in ts] for ts in input_list]

        padded_batch = torch.LongTensor(input_list)

        input_mask = torch.ne(padded_batch, self.word2idx[PAD_WORD]).type(torch.FloatTensor)

        return padded_batch, input_list_lens, input_mask

    def collate_fn_common(self, batches, trg=None, trg_oov=None):
        # source with oov words replaced by <unk>
        src = [b['src'] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [b['src_oov'] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]

        # b['src_str'] is a word_list for source text
        # b['trg_str'] is a list of word list
        src_str = [b['src_str'] for b in batches]
        trg_str = [b['trg_str'] for b in batches]

        batch_size = len(src)
        original_indices = list(range(batch_size))

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        src_lens = [len(i) for i in src]

        seq_pairs = sorted(zip(src_lens, src, src_oov, oov_lists, src_str, trg_str, original_indices),
                           key=lambda p: p[0], reverse=True)
        _, src, src_oov, oov_lists, src_str, trg_str, original_indices = zip(
            *seq_pairs)

        if self.load_train:
            seq_pairs = sorted(zip(src_lens, trg, trg_oov), key=lambda p: p[0], reverse=True)
            _, trg, trg_oov = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        src_oov, _, _ = self._pad(src_oov)

        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        src_oov = src_oov.to(self.device)

        if self.load_train:
            if self.fix_kp_num_len:
                trg, trg_lens, trg_mask = self._pad2d(trg)
                trg_oov, _, _ = self._pad2d(trg_oov)
            else:
                trg, trg_lens, trg_mask = self._pad(trg)
                trg_oov, _, _ = self._pad(trg_oov)

            trg = trg.to(self.device)
            trg_mask = trg_mask.to(self.device)
            trg_oov = trg_oov.to(self.device)
        else:
            trg_lens, trg_mask = None, None

        return src, src_lens, src_mask, src_oov, oov_lists, src_str, \
               trg_str, trg, trg_oov, trg_lens, trg_mask, original_indices

    def collate_fn_one2one(self, batches):
        if self.load_train:
            trg = [b['trg'] + [self.word2idx[EOS_WORD]] for b in batches]
            trg_oov = [b['trg_copy'] + [self.word2idx[EOS_WORD]] for b in batches]
            return self.collate_fn_common(batches, trg, trg_oov)
        else:
            return self.collate_fn_common(batches)

    def collate_fn_one2seq(self, batches):
        if self.load_train:
            trg = []
            trg_oov = []
            for b in batches:
                trg_concat = []
                trg_oov_concat = []
                trg_size = len(b['trg'])
                assert len(b['trg']) == len(b['trg_copy'])
                for trg_idx, (trg_phase, trg_phase_oov) in enumerate(zip(b['trg'], b['trg_copy'])):
                    # ignore the <peos> word if it exists
                    if self.word2idx[PEOS_WORD] in trg_phase:
                        continue
                    # if this is the last keyphrase, end with <eos>
                    if trg_idx == trg_size - 1:
                        trg_concat += trg_phase + [self.word2idx[EOS_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[EOS_WORD]]
                    else:
                        # trg_concat = [target_1] + [sep] + [target_2] + [sep] + ...
                        trg_concat += trg_phase + [self.word2idx[SEP_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[SEP_WORD]]
                trg.append(trg_concat)
                trg_oov.append(trg_oov_concat)

            return self.collate_fn_common(batches, trg, trg_oov)
        else:
            return self.collate_fn_common(batches)

    def collate_fn_fixed_tgt(self, batches):
        if self.load_train:
            if self.seperate_pre_ab:
                trg = []
                trg_oov = []
                for b in batches:
                    targets = [t for t in b['trg'] if len(t) <= (self.max_kp_len - 1)]
                    oov_targets = [t for t in b['trg_copy'] if len(t) <= (self.max_kp_len - 1)]
                    assert [self.word2idx[PEOS_WORD]] in targets, \
                        "the original training keyphrases must be seperated by <peos> !"
                    peos_idx = targets.index([self.word2idx[PEOS_WORD]])

                    present_targets = targets[:peos_idx][:self.max_kp_num // 2]
                    absent_targets = targets[peos_idx + 1:][:self.max_kp_num // 2]
                    present_targets_oov = oov_targets[:peos_idx][:self.max_kp_num // 2]
                    absent_targets_oov = oov_targets[peos_idx + 1:][:self.max_kp_num // 2]

                    # padding present keyphrase
                    present_targets = [
                        t + [self.word2idx[EOS_WORD]] + [self.word2idx[PAD_WORD]] * (self.max_kp_len - len(t) - 1)
                        for t in present_targets]
                    present_targets_oov = [
                        t + [self.word2idx[EOS_WORD]] + [self.word2idx[PAD_WORD]] * (self.max_kp_len - len(t) - 1)
                        for t in present_targets_oov]
                    extra_present_targets = [[self.word2idx[NULL_WORD]] + [self.word2idx[PAD_WORD]] * (
                            self.max_kp_len - 1)] * (self.max_kp_num // 2 - len(present_targets))

                    # padding absent keyphrase
                    absent_targets = [
                        t + [self.word2idx[EOS_WORD]] + [self.word2idx[PAD_WORD]] * (self.max_kp_len - len(t) - 1)
                        for t in absent_targets]
                    absent_targets_oov = [
                        t + [self.word2idx[EOS_WORD]] + [self.word2idx[PAD_WORD]] * (self.max_kp_len - len(t) - 1)
                        for t in absent_targets_oov]
                    extra_absent_targets = [[self.word2idx[NULL_WORD]] + [self.word2idx[PAD_WORD]] * (
                            self.max_kp_len - 1)] * (self.max_kp_num // 2 - len(absent_targets))

                    trg.append(present_targets + extra_present_targets + absent_targets + extra_absent_targets)
                    trg_oov.append(
                        present_targets_oov + extra_present_targets + absent_targets_oov + extra_absent_targets)
            else:
                trg = []
                trg_oov = []
                for b in batches:
                    targets = [t + [self.word2idx[EOS_WORD]] + [self.word2idx[PAD_WORD]] * (
                            self.max_kp_len - len(t) - 1)
                               for t in b['trg'] if len(t) <= (self.max_kp_len - 1)][:self.max_kp_num]
                    oov_targets = [t + [self.word2idx[EOS_WORD]] + [self.word2idx[PAD_WORD]] * (
                            self.max_kp_len - len(t) - 1)
                                   for t in b['trg_copy'] if len(t) <= (self.max_kp_len - 1)][:self.max_kp_num]

                    extra_targets = [[self.word2idx[NULL_WORD]] + [self.word2idx[PAD_WORD]] * (
                            self.max_kp_len - 1)] * (self.max_kp_num - len(targets))
                    trg.append(targets + extra_targets)
                    trg_oov.append(oov_targets + extra_targets)

            return self.collate_fn_common(batches, trg, trg_oov)
        else:
            return self.collate_fn_common(batches)


def build_interactive_predict_dataset(tokenized_src, opt, mode="one2many", include_original=True):
    # build a dummy trg list, and then combine it with src, and pass it to the build_dataset method
    num_lines = len(tokenized_src)
    tokenized_trg = [['.']] * num_lines  # create a dummy tokenized_trg
    tokenized_src_trg_pairs = list(zip(tokenized_src, tokenized_trg))
    return build_dataset(tokenized_src_trg_pairs, opt, mode=mode, include_original=include_original)


def build_dataset(src_trgs_pairs, opt, mode='one2one', include_original=True):
    '''
    Standard process for copy model
    :param mode: one2one or one2many
    :param include_original: keep the original texts of source and target
    :return:
    '''
    word2idx = opt.vocab['word2idx']
    return_examples = []
    oov_target = 0
    max_oov_len = 0
    max_oov_sent = ''

    for idx, (source, targets) in enumerate(src_trgs_pairs):
        # if w's id is larger than opt.vocab_size, replace with <unk>
        src = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in source]

        src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2idx, opt.vocab_size, opt.max_unk_words)

        examples = []  # for one-to-many
        for target in targets:
            example = {}
            if include_original:
                example['src_str'] = source
                example['trg_str'] = target
            example['src'] = src
            example['src_oov'] = src_oov
            example['oov_dict'] = oov_dict
            example['oov_list'] = oov_list
            if len(oov_list) > max_oov_len:
                max_oov_len = len(oov_list)
                max_oov_sent = source

            trg = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD]
                   for w in target]
            example['trg'] = trg

            # oov words are replaced with new index
            trg_copy = []
            for w in target:
                if w in word2idx and word2idx[w] < opt.vocab_size:
                    trg_copy.append(word2idx[w])
                elif w in oov_dict:
                    trg_copy.append(oov_dict[w])
                else:
                    trg_copy.append(word2idx[UNK_WORD])
            example['trg_copy'] = trg_copy

            if any([w >= opt.vocab_size for w in trg_copy]):
                oov_target += 1

            if mode == 'one2one':
                return_examples.append(example)
            else:
                examples.append(example)

        if mode == 'one2many' and len(examples) > 0:
            o2m_example = {}
            keys = examples[0].keys()
            for key in keys:
                if key.startswith('src') or key.startswith('oov') or key.startswith('title'):
                    o2m_example[key] = examples[0][key]
                else:
                    o2m_example[key] = [e[key] for e in examples]
            if include_original:
                assert len(o2m_example['src']) == len(o2m_example['src_oov']) == len(o2m_example['src_str'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy']) == len(o2m_example['trg_str'])
            else:
                assert len(o2m_example['src']) == len(o2m_example['src_oov'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy'])

            return_examples.append(o2m_example)

    logging.info('Find #(oov_target)/#(all) = %d/%d' % (oov_target, len(return_examples)))
    logging.info('Find max_oov_len = %d' % max_oov_len)
    logging.info('max_oov sentence: %s' % str(max_oov_sent))

    return return_examples


def extend_vocab_OOV(source_words, word2idx, vocab_size, max_unk_words):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
    """
    src_oov = []
    oov_dict = {}
    for w in source_words:
        if w in word2idx and word2idx[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
            src_oov.append(word2idx[w])
        else:
            if len(oov_dict) < max_unk_words:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                word_id = oov_dict.get(w, len(oov_dict) + vocab_size)
                oov_dict[w] = word_id
                src_oov.append(word_id)
            else:
                # exceeds the maximum number of acceptable oov words, replace it with <unk>
                word_id = word2idx[UNK_WORD]
                src_oov.append(word_id)

    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x: x[1])]
    return src_oov, oov_dict, oov_list
