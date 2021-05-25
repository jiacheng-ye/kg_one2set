import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn

import pykp.utils.io as io
from inference.evaluate import evaluate_loss
from pykp.utils.label_assign import hungarian_assign
from pykp.utils.masked_loss import masked_cross_entropy
from utils.functions import time_since
from utils.report import export_train_and_valid_loss
from utils.statistics import LossStatistics

EPS = 1e-8


def train_model(model, optimizer, train_data_loader, valid_data_loader, opt):
    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    num_stop_dropping = 0

    model.train()
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        if early_stop_flag:
            break
        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            batch_loss_stat = train_one_batch(batch, model, optimizer, opt)
            report_train_loss_statistics.update(batch_loss_stat)
            total_train_loss_statistics.update(batch_loss_stat)

            if total_batch % opt.report_every == 0:
                current_train_ppl = report_train_loss_statistics.ppl()
                current_train_loss = report_train_loss_statistics.xent()
                logging.info(
                    "Epoch %d; batch: %d; total batch: %d，avg training ppl: %.3f, loss: %.3f" % (epoch, batch_i,
                                                                                                 total_batch,
                                                                                                 current_train_ppl,
                                                                                                 current_train_loss))
            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and
                         total_batch % opt.checkpoint_interval == 0):
                    valid_loss_stat = evaluate_loss(valid_data_loader, model, opt)
                    model.train()

                    current_valid_loss = valid_loss_stat.xent()
                    current_valid_ppl = valid_loss_stat.ppl()
                    logging.info("Enter check point!")

                    current_train_ppl = report_train_loss_statistics.ppl()
                    current_train_loss = report_train_loss_statistics.xent()

                    # debug
                    if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (
                                epoch, batch_i, total_batch))
                        exit()

                    if current_valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
                        logging.info("Valid loss drops")
                        sys.stdout.flush()
                        best_valid_loss = current_valid_loss
                        best_valid_ppl = current_valid_ppl
                        num_stop_dropping = 0

                        check_pt_model_path = os.path.join(opt.model_path, 'best_model.pt')
                        torch.save(  # save model parameters
                            model.state_dict(),
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving checkpoint to %s' % check_pt_model_path)
                    else:
                        num_stop_dropping += 1
                        logging.info("Valid loss does not drop, patience: %d/%d" % (
                            num_stop_dropping, opt.early_stop_tolerance))

                        # decay the learning rate by a factor
                        for i, param_group in enumerate(optimizer.param_groups):
                            old_lr = float(param_group['lr'])
                            new_lr = old_lr * opt.learning_rate_decay
                            if old_lr - new_lr > EPS:
                                param_group['lr'] = new_lr

                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        ' * avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                            current_train_ppl, current_valid_ppl, best_valid_ppl))
                    logging.info(
                        ' * avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                            current_train_loss, current_valid_loss, best_valid_loss))

                    report_train_ppl.append(current_train_ppl)
                    report_valid_ppl.append(current_valid_ppl)
                    report_train_loss.append(current_train_loss)
                    report_valid_loss.append(current_valid_loss)

                    if num_stop_dropping >= opt.early_stop_tolerance:
                        logging.info(
                            'Have not increased for %d check points, early stop training' % num_stop_dropping)
                        early_stop_flag = True
                        break
                    report_train_loss_statistics.clear()

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl,
                                opt.checkpoint_interval, train_valid_curve_path)


def train_one_batch(batch, model, optimizer, opt):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
    trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = batch

    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
    batch_size = src.size(0)
    word2idx = opt.vocab['word2idx']
    target = trg_oov if opt.copy_attention else trg

    optimizer.zero_grad()
    start_time = time.time()
    if opt.fix_kp_num_len:
        y_t_init = target.new_ones(batch_size, opt.max_kp_num, 1) * word2idx[io.BOS_WORD]
        if opt.set_loss:  # K-step target assignment
            model.eval()
            with torch.no_grad():
                memory_bank = model.encoder(src, src_lens, src_mask)
                state = model.decoder.init_state(memory_bank, src_mask)
                control_embed = model.decoder.forward_seg(state)

                input_tokens = src.new_zeros(batch_size, opt.max_kp_num, opt.assign_steps + 1)
                decoder_dists = []
                input_tokens[:, :, 0] = word2idx[io.BOS_WORD]
                for t in range(1, opt.assign_steps + 1):
                    decoder_inputs = input_tokens[:, :, :t]
                    decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(opt.vocab_size - 1),
                                                                word2idx[io.UNK_WORD])

                    decoder_dist, _ = model.decoder(decoder_inputs, state, src_oov, max_num_oov, control_embed)
                    input_tokens[:, :, t] = decoder_dist.argmax(-1)
                    decoder_dists.append(decoder_dist.reshape(batch_size, opt.max_kp_num, 1, -1))

                decoder_dists = torch.cat(decoder_dists, -2)

                if opt.seperate_pre_ab:
                    mid_idx = opt.max_kp_num // 2
                    pre_reorder_index = hungarian_assign(decoder_dists[:, :mid_idx],
                                                         target[:, :mid_idx, :opt.assign_steps],
                                                         ignore_indices=[word2idx[io.NULL_WORD],
                                                                         word2idx[io.PAD_WORD]])
                    target[:, :mid_idx] = target[:, :mid_idx][pre_reorder_index]
                    trg_mask[:, :mid_idx] = trg_mask[:, :mid_idx][pre_reorder_index]

                    ab_reorder_index = hungarian_assign(decoder_dists[:, mid_idx:],
                                                        target[:, mid_idx:, :opt.assign_steps],
                                                        ignore_indices=[word2idx[io.NULL_WORD],
                                                                        word2idx[io.PAD_WORD]])
                    target[:, mid_idx:] = target[:, mid_idx:][ab_reorder_index]
                    trg_mask[:, mid_idx:] = trg_mask[:, mid_idx:][ab_reorder_index]
                else:
                    reorder_index = hungarian_assign(decoder_dists, target[:, :, :opt.assign_steps],
                                                     [word2idx[io.NULL_WORD],
                                                      word2idx[io.PAD_WORD]])
                    target = target[reorder_index]
                    trg_mask = trg_mask[reorder_index]
            model.train()

        input_tgt = torch.cat([y_t_init, target[:, :, :-1]], dim=-1)
        memory_bank = model.encoder(src, src_lens, src_mask)
        state = model.decoder.init_state(memory_bank, src_mask)
        control_embed = model.decoder.forward_seg(state)

        input_tgt = input_tgt.masked_fill(input_tgt.gt(opt.vocab_size - 1), word2idx[io.UNK_WORD])
        decoder_dist, attention_dist = model.decoder(input_tgt, state, src_oov, max_num_oov, control_embed)
    else:
        y_t_init = trg.new_ones(batch_size, 1) * word2idx[io.BOS_WORD]  # [batch_size, 1]
        input_tgt = torch.cat([y_t_init, trg[:, :-1]], dim=-1)
        memory_bank = model.encoder(src, src_lens, src_mask)
        state = model.decoder.init_state(memory_bank, src_mask)
        decoder_dist, attention_dist = model.decoder(input_tgt, state, src_oov, max_num_oov)

    forward_time = time_since(start_time)
    start_time = time.time()
    if opt.fix_kp_num_len:
        if opt.seperate_pre_ab:
            mid_idx = opt.max_kp_num // 2
            pre_loss = masked_cross_entropy(
                decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)[:, :mid_idx]\
                    .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                target[:, :mid_idx].reshape(batch_size, -1),
                trg_mask[:, :mid_idx].reshape(batch_size, -1),
                loss_scales=[opt.loss_scale_pre],
                scale_indices=[word2idx[io.NULL_WORD]])
            ab_loss = masked_cross_entropy(
                decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)[:, mid_idx:]
                    .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                target[:, mid_idx:].reshape(batch_size, -1),
                trg_mask[:, mid_idx:].reshape(batch_size, -1),
                loss_scales=[opt.loss_scale_ab],
                scale_indices=[word2idx[io.NULL_WORD]])
            loss = pre_loss + ab_loss
        else:
            loss = masked_cross_entropy(decoder_dist, target.reshape(batch_size, -1), trg_mask.reshape(batch_size, -1),
                                        loss_scales=[opt.loss_scale], scale_indices=[word2idx[io.NULL_WORD]])
    else:
        loss = masked_cross_entropy(decoder_dist, target, trg_mask)
    loss_compute_time = time_since(start_time)

    total_trg_tokens = trg_mask.sum().item()
    total_trg_sents = src.size(0)
    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        normalization = total_trg_sents
    else:
        raise ValueError('The type of loss normalization is invalid.')
    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    total_loss = loss.div(normalization)

    total_loss.backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

    optimizer.step()
    stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time,
                          loss_compute_time=loss_compute_time, backward_time=backward_time)
    return stat
