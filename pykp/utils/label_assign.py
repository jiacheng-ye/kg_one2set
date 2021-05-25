import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def hungarian_assign(decode_dist, target, ignore_indices, random=False):
    '''

    :param decode_dist: (batch_size, max_kp_num, kp_len, vocab_size)
    :param target: (batch_size, max_kp_num, kp_len)
    :return:
    '''

    batch_size, max_kp_num, kp_len = target.size()
    reorder_rows = torch.arange(batch_size)[..., None]
    if random:
        reorder_cols = np.concatenate([np.random.permutation(max_kp_num).reshape(1, -1) for _ in range(batch_size)], axis=0)
    else:
        score_mask = target.new_zeros(target.size()).bool()
        for i in ignore_indices:
            score_mask |= (target == i)
        score_mask = score_mask.unsqueeze(1)  # (batch_size, 1, max_kp_num, kp_len)

        score = decode_dist.new_zeros(batch_size, max_kp_num, max_kp_num, kp_len)
        for b in range(batch_size):
            for l in range(kp_len):
                score[b, :, :, l] = decode_dist[b, :, l][:, target[b, :, l]]
        score = score.masked_fill(score_mask, 0)
        score = score.sum(-1)  # batch_size, max_kp_num, max_kp_num

        reorder_cols = []
        for b in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(score[b].detach().cpu().numpy(), maximize=True)
            reorder_cols.append(col_ind.reshape(1, -1))
            # total_score += sum(score[b][row_ind, col_ind])
        reorder_cols = np.concatenate(reorder_cols, axis=0)
    return tuple([reorder_rows, reorder_cols])


if __name__ == '__main__':
    '''
     :param decoder_dist: (batch_size, max_kp_num * max_kp_len, vocab_size)
     :param trg:  (batch_size, max_kp_num, max_kp_len)
     :param trg_mask:  (batch_size, max_kp_num, max_kp_len)
     :return:
    '''
    # torch.manual_seed(5)
    torch.manual_seed(3)

    decoder_dist = torch.rand((1, 3, 2, 4)).softmax(-1)
    target = torch.randint(3, (1, 3, 2))
    targets = torch.randint(3, (1, 3, 2))
    trg_mask = torch.randint(2, (1, 3, 2))
    ignore_idx = 0

    print("decoder_dist: \n", decoder_dist)
    print("target: \n", target)
    print("targets: \n", targets)
    print("trg_mask： \n", trg_mask)
    reorder_index = hungarian_assign(decoder_dist, target, [0])
    print("new new_decoder_dist: \n", decoder_dist[reorder_index])
    print("new new_targets: \n", targets[reorder_index])
    print("new new_trg_mask: \n", trg_mask[reorder_index])
