import torch
import numpy as np


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    sinusoid的embedding，其中position的表示中，偶数维(0,2,4,...)是sin, 奇数(1,3,5...)是cos
    :param int n_position: 一共多少个position
    :param int d_hid: 多少维度，需要为偶数
    :param padding_idx:
    :return: torch.FloatTensor, shape为n_position x d_hid
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

