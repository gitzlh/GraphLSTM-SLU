import numpy as np
import torch
import torch.utils.data
import Constants
from allennlp.modules.elmo import Elmo, batch_to_ids

torch.manual_seed(Constants.SEED)
torch.cuda.manual_seed(Constants.SEED)


def paired_collate_fn(insts):
    '''
    :param insts:a list of length batch_size,each element is ([],[])
    :return: src_seq,tgt_seq
    '''
    src_insts, src_char_insts, tgt_insts = list(zip(*insts))
    src_seq = collate_fn(src_insts)
    src_char = collate_fn(src_char_insts, is_char=True)
    tgt_seq = collate_fn(tgt_insts)
    return (src_seq, src_char, tgt_seq)


def collate_fn(insts, is_char=False):
    '''
     Pad the instance to the max seq length in batch
     usage:passed as a parameter to dataloader to create batch
    '''
    if is_char:
        batch_seq = [inst for inst in insts]
        batch_seq = batch_to_ids(batch_seq)
    else:
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([inst + [Constants.PAD] * (max_len - len(inst)) for inst in insts])
        batch_seq = torch.LongTensor(batch_seq)
    return batch_seq


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_word2idx, tgt_word2idx, src_insts, src_char_insts,
                 tgt_insts=None):
        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))
        src_idx2word = {idx: word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts
        self._src_char_insts = src_char_insts

        tgt_idx2word = {idx: word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src_insts[idx], \
                   self._src_char_insts[idx], \
                   self._tgt_insts[idx]
        return self._src_insts[idx]
