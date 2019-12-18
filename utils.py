"""
Utility functions for the ESIM model.
"""
# Aurelien Coet, 2018.
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import Constants
import numpy as np
import logging
import os
import sys

import torch
import numpy as np


def input_context(in_tensor):
    '''
    :param in_tensor: (l,b,d)
    :return: (l,b,3d)
    '''
    out_tensor = torch.cuda.FloatTensor(in_tensor.size(0), in_tensor.size(1), in_tensor.size(2) * 3)
    for i, embs in enumerate(in_tensor):
        if i == 0:
            step_tensor = torch.cat([in_tensor[i], in_tensor[i], in_tensor[i + 1]],dim=-1)
        elif i == in_tensor.size(0) - 1:
            step_tensor = torch.cat([in_tensor[i - 1], in_tensor[i], in_tensor[i]],dim=-1)
        else:
            step_tensor = torch.cat([in_tensor[i - 1], in_tensor[i], in_tensor[i + 1]],dim=-1)
        out_tensor[i] = step_tensor
    return out_tensor


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    '''
    :param batch: (b,l,h)
    :param sequences_lengths: (b)
    :param descending: Bool
    :return: sorted_batch:(b,l,h)
             sorted_seq_lens:(b)
             sorting index:(b)
             restoration_index:(b)
    '''
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def masked_softmax(vector, mask, dim=-1):
    '''
    :param vector: (b,la,lb)
    :param mask: (b,lb) or (b,la,lb)
    :param dim: int
    :return: (b,la,lb)
    '''
    if mask is None:
        result = F.softmax(vector, dim=dim)
    else:
        mask = mask.byte()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector.masked_fill(mask, -np.inf)
        result = F.softmax(vector, dim=dim)
    return result


def get_len(tensor):
    '''
    :param tensor:(b,l)
    :return: (b)
    '''
    mask = tensor.ne(Constants.PAD)  # (b,la)
    return mask.sum(dim=-1)


def weighted_sum(matrix, attention):
    """
    :param matrix: (b,lb,h)
    :param attention: (b,la,lb) or (b,lb)
    :return: (b,la,h) or (b,h)
    """
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)
