import os
import Constants

os.environ["CUDA_VISIBLE_DEVICES"] = Constants.GPU
import argparse
import math
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import Constants
from dataset import TranslationDataset, paired_collate_fn
import functools
import operator
import numpy as np
import logging
import os
import random
from models.SLSTM import sLSTM
from evaluate import *
from utils import get_len
import sys

logging.disable(sys.maxsize)  # Python 3
np.random.seed(Constants.SEED)
torch.manual_seed(Constants.SEED)
torch.cuda.manual_seed(Constants.SEED)
random.seed(Constants.SEED)
torch.backends.cudnn.deterministic = True


def cal_correct(pred, gold):
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return n_correct


def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    slot_total = 0
    slot_correct = 0
    intent_total = 0
    intent_correct = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        src_seq, src_char, tgt_seq = map(lambda x: x.to(device), batch)  # [B,max_len]
        slot_gold = tgt_seq[:, 1:].transpose(0, 1).contiguous().view(-1)
        intent_gold = tgt_seq[:, 0].view(-1)
        intent_gold[intent_gold != Constants.UNK] -= Constants.TAGS
        src_seq = src_seq[:, 1:]
        src_char = src_char[:, 1:]

        optimizer.zero_grad()
        slot_logit, intent_logit = model(src_seq.transpose(0, 1), src_char, )  # [B*C]
        slot_loss = F.cross_entropy(slot_logit, slot_gold, ignore_index=Constants.PAD, reduction='sum')
        intent_loss = F.cross_entropy(intent_logit, intent_gold, reduction='sum')
        batch_loss = slot_loss + intent_loss
        batch_slot_correct = cal_correct(slot_logit, slot_gold)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        slot_correct += batch_slot_correct
        non_pad_mask = slot_gold.ne(Constants.PAD)
        batch_word = non_pad_mask.sum().item()
        slot_total += batch_word

    loss_per_word = total_loss / slot_total
    slot_accuracy = slot_correct / slot_total
    return loss_per_word, slot_accuracy


def eval_model(model, validation_data, device, opt):
    model.eval()
    data = torch.load(opt.data)
    tgt_word2idx = data['dict']['tgt']
    tgt_idx2word = {idx: word for word, idx in tgt_word2idx.items()}
    intent_word2idx = data['dict']['intent']
    intent_idx2word = {idx: word for word, idx in intent_word2idx.items()}
    slot_preds = []
    slot_golds = []
    intent_preds = []
    intent_golds = []
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (computing F1) ', leave=False):
            src_seq, src_char, tgt_seq = map(lambda x: x.to(device), batch)  # [B,max_len]
            src_seq = src_seq[:, 1:]
            src_char = src_char[:, 1:]
            slot_gold = tgt_seq[:, 1:]
            max_len = src_seq.size(1)
            src_len = torch.cuda.LongTensor(np.array(get_len(src_seq.cpu()))).unsqueeze(1)

            intent_gold = tgt_seq[:, 0].view(-1)
            intent_gold[intent_gold != Constants.UNK] -= Constants.TAGS

            slot_logit, intent_logit = model(src_seq.transpose(0, 1), src_char)  # [B*C]
            intent_pred = intent_logit.max(1)[1]
            slot_pred = slot_logit.max(1)[1].view(max_len, -1).transpose(0, 1)
            slot_pred = [
                [tgt_idx2word.get(elem.item(), Constants.UNK_WORD) for elem in elems if elem.item() != Constants.PAD]
                for elems in slot_pred]
            slot_gold = [
                [tgt_idx2word.get(elem.item(), Constants.UNK_WORD) for elem in elems if elem.item() != Constants.PAD]
                for elems in slot_gold]

            intent_pred = [intent_idx2word.get(elem.item() + Constants.TAGS, Constants.UNK_WORD) for elem in
                           intent_pred]
            intent_gold = [intent_idx2word.get(elem.item() + Constants.TAGS, Constants.UNK_WORD) for elem in
                           intent_gold]

            for i in range(len(slot_gold)):
                slot_pred[i] = slot_pred[i][:len(slot_gold[i])]
            slot_preds += slot_pred
            slot_golds += slot_gold
            intent_preds += intent_pred
            intent_golds += intent_gold

    lines = ''
    for intent_pred, slot_pred in zip(intent_preds, slot_preds):
        line = intent_pred + ' ' + ' '.join(slot_pred)
        lines += line + '\n'
    with open('pred.txt', 'w') as f:
        f.write(lines)

    lines = ''
    for intent_gold, slot_gold in zip(intent_golds, slot_golds):
        line = intent_gold + ' ' + ' '.join(slot_gold)
        lines += line + '\n'
    with open('gold.txt', 'w') as f:
        f.write(lines.strip())

    f1, precision, recall = computeF1Score(slot_golds, slot_preds)
    intent_acc, sent_acc = get_sent_acc('gold.txt', 'pred.txt')

    return f1, precision, recall, intent_acc, sent_acc


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in training phase'''

    model.eval()

    total_loss = 0
    slot_total = 0
    slot_correct = 0
    intent_total = 0
    intent_correct = 0

    for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation)   ', leave=False):
        src_seq, src_char, tgt_seq = map(lambda x: x.to(device), batch)  # [B,max_len]
        src_seq = src_seq[:, 1:]
        src_char = src_char[:, 1:]
        slot_gold = tgt_seq[:, 1:].transpose(0, 1).contiguous().view(-1)
        intent_gold = tgt_seq[:, 0].view(-1)
        intent_gold[intent_gold != Constants.UNK] -= Constants.TAGS
        slot_logit, intent_logit = model(src_seq.transpose(0, 1), src_char)  # [B*C]
        slot_loss = F.cross_entropy(slot_logit, slot_gold, ignore_index=Constants.PAD, reduction='sum')
        intent_loss = F.cross_entropy(intent_logit, intent_gold, reduction='sum')
        batch_loss = slot_loss + intent_loss
        batch_slot_correct = cal_correct(slot_logit, slot_gold)
        total_loss += batch_loss.item()
        slot_correct += batch_slot_correct
        non_pad_mask = slot_gold.ne(Constants.PAD)
        batch_word = non_pad_mask.sum().item()
        slot_total += batch_word

    loss_per_word = total_loss / slot_total
    slot_accuracy = slot_correct / slot_total
    return loss_per_word, slot_accuracy


def train(model, training_data, validation_data, test_data, optimizer, device, opt):
    ''' Start training '''

    valid_accus = []
    f1s = []
    sent_accs = []
    best_slot = 0
    best_intent = 0
    best_sent = 0
    per_epoch = 5
    for epoch_i in range(opt.epoch):
        if epoch_i % per_epoch == 0:
            print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, device,
                                             )  # the train_loss is per word loss!
        if epoch_i % per_epoch == 0:
            print('  - (Training)   loss: {loss: 8.5f}  ' \
                  'elapse: {elapse:3.3f} min'.format(loss=train_loss,
                                                     elapse=(time.time() - start) / 60))
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)

        if epoch_i % per_epoch == 0:
            print('  - (Validation) loss: {loss: 8.5f}  ' \
                  'elapse: {elapse:3.3f} min'.format(loss=valid_loss,
                                                     elapse=(time.time() - start) / 60))
        start = time.time()
        test_loss, test_accu = eval_epoch(model, test_data, device)

        if epoch_i % per_epoch == 0:
            print('  - (Test)       loss: {loss: 8.5f}  ' \
                  'elapse: {elapse:3.3f} min'.format(loss=test_loss,
                                                     elapse=(time.time() - start) / 60))
        dev_f1, dev_precision, dev_recall, dev_intent_acc, dev_sent_acc = eval_model(model, validation_data, device,
                                                                                     opt)
        f1, precision, recall, intent_acc, sent_acc = eval_model(model, test_data, device, opt)
        if epoch_i % per_epoch == 0:
            print('  Test  F1: %.3f   Intent:%.3f  Sent_acc:%.3f' % ( f1, intent_acc,sent_acc))
        # sent_accs.append(dev_sent_acc) # using sent_acc to chose model
        # if dev_sent_acc >= max(sent_accs):
        f1s += [dev_f1]
        if dev_f1 >= max(f1s):
            best_slot = f1
            best_intent = intent_acc
            best_sent = sent_acc
        if epoch_i % per_epoch == 0:
            print('  - Current best $$  F1: %.3f, Intent acc: %.3f, Sent acc :%.3f' % (
                best_slot, best_intent, best_sent))
        valid_accus.append(valid_accu)
        model_state_dict = model.state_dict()
        checkpoint = {'model': model_state_dict, 'settings': opt, 'epoch': epoch_i}
        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * f1)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model
                if dev_f1 >= max(f1s):
                    torch.save(checkpoint, model_name)


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=150)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-d_model', type=int, default=150)
    parser.add_argument('-d_word', type=int, default=150)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-data', default='data/snips.pt', required=False)
    parser.add_argument('-save_model', default='snips-tmp.chkpt')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-restore_model', default='snips.chkpt')
    # parser.add_argument('-restore_model', default=None)
    parser.add_argument('-no_cuda', default=False, action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    # opt.d_word_vec = opt.d_model

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len
    training_data, validation_data, test_data = prepare_dataloaders(data, opt)
    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size
    # ========= Preparing Model =========#
    print(opt)
    device = torch.device('cuda' if opt.cuda else 'cpu')
    embeddings = None
    opt.n_intent = len(data['dict']['intent'])
    opt.n_tgt = len(data['dict']['tgt'])
    model = sLSTM(d_word=opt.d_word, d_hidden=opt.d_model, n_src_vocab=opt.src_vocab_size,
                  n_tgt_vocab=opt.tgt_vocab_size, n_intent=opt.n_intent, dropout=opt.dropout,
                  embeddings=embeddings).to(device)
    num_parameters = sum(functools.reduce(operator.mul, parameter.size(), 1)
                         for parameter in model.parameters())
    print("[INFO] Total parameter number: ", num_parameters)
    if opt.restore_model:
        checkpoint = torch.load(opt.restore_model)
        model.load_state_dict(checkpoint['model'])
        print('[Info] Old Trained model state loaded.')

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09)
    # optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001,momentum=0.9)

    train(model, training_data, validation_data, test_data, optimizer, device, opt)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(src_word2idx=data['dict']['src'], tgt_word2idx=data['dict']['tgt'],
                           src_insts=data['train']['src'],
                           src_char_insts=data['train']['src_char'],
                           tgt_insts=data['train']['tgt'], ), num_workers=2, batch_size=opt.batch_size,
        collate_fn=paired_collate_fn, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(src_word2idx=data['dict']['src'], tgt_word2idx=data['dict']['tgt'],
                           src_insts=data['valid']['src'],
                           src_char_insts=data['valid']['src_char'],
                           tgt_insts=data['valid']['tgt']), num_workers=2, batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(src_word2idx=data['dict']['src'], tgt_word2idx=data['dict']['tgt'],
                           src_insts=data['test']['src'],
                           src_char_insts=data['test']['src_char'],
                           tgt_insts=data['test']['tgt']), num_workers=2, batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    main()
