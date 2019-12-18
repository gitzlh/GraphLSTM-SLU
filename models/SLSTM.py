import os
import Constants

os.environ["CUDA_VISIBLE_DEVICES"] = Constants.GPU
from utils import *
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder


class sLSTMCell(nn.Module):
    '''
    Adapted from: https://github.com/WildeLau/S-LSTM_pytorch
    '''

    def __init__(self, d_word, d_hidden, n_windows,
                 n_sent_nodes, bias, batch_first,
                 init_method='normal'):
        super().__init__()
        self.d_input = d_word
        self.d_hidden = d_hidden
        self.n_windows = n_windows
        self.num_g = n_sent_nodes
        self.initial_method = init_method
        self.bias = bias
        self.batch_first = batch_first
        self.lens_dim = 1 if batch_first is True else 0
        self._all_gate_weights = []
        word_gate_dict = dict(
            [('input_gate', 'i'), ('left_forget_gate', 'l'),
             ('right_forget_gate', 'r'), ('forget_gate', 'f'),
             ('sentence_forget_gate', 's'), ('output_gate', 'o'),
             ('recurrent_input', 'u')])

        for (gate_name, gate_tag) in word_gate_dict.items():
            w_w = nn.Parameter(torch.Tensor(d_hidden,
                                            (n_windows * 2 + 1) * d_hidden))
            w_u = nn.Parameter(torch.Tensor(d_hidden, d_word))
            w_v = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
            w_b = nn.Parameter(torch.Tensor(d_hidden))

            gate_params = (w_w, w_u, w_v, w_b)
            param_names = ['w_w{}', 'w_u{}', 'w_v{}', 'w_b{}']
            param_names = [x.format(gate_tag) for x in param_names]  # {
            for name, param in zip(param_names, gate_params):
                setattr(self, name, param)  # self.name = param
            self._all_gate_weights.append(param_names)

        sentence_gate_dict = dict(
            [('sentence_forget_gate', 'g'), ('word_forget_gate', 'f'),
             ('output_gate', 'o')])

        for (gate_name, gate_tag) in sentence_gate_dict.items():
            s_w = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
            s_u = nn.Parameter(torch.Tensor(d_hidden, d_hidden))
            s_b = nn.Parameter(torch.Tensor(d_hidden))
            gate_params = (s_w, s_u, s_b)
            param_names = ['s_w{}', 's_u{}', 's_b{}']
            param_names = [x.format(gate_tag) for x in param_names]
            for name, param in zip(param_names, gate_params):
                setattr(self, name, param)
            self._all_gate_weights.append(param_names)
        self.reset_parameters(self.initial_method)

    def reset_parameters(self, init_method):
        if init_method is 'normal':
            std = 0.1
            for weight in self.parameters():
                weight.data.normal_(mean=0.0, std=std)
        else:  # uniform
            stdv = 1.0 / math.sqrt(self.d_hidden)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def sequence_mask(self, size, length):  # ???
        mask = torch.LongTensor(range(size[0])).view(size[0], 1).cuda()  # (l,1)
        length = length.squeeze(dim=1)  # (b)
        result = (mask >= length).unsqueeze(dim=2)  # (l,b,1)
        return result

    def in_window_context(self, hx, window_size=1, average=False):
        slices = torch.unbind(hx, dim=0)  # torch.size([18,32,256]) -> ([32,256]) * 18
        zeros = torch.unbind(torch.zeros_like(hx), dim=0)

        context_l = [torch.stack(zeros[:i] + slices[:len(slices) - i], dim=0)
                     for i in range(window_size, 0, -1)]
        context_l.append(hx)
        context_r = [torch.stack(slices[i + 1: len(slices)] + zeros[:i + 1], dim=0)
                     for i in range(0, window_size)]

        context = context_l + context_r
        return torch.stack(context).mean(dim=0) if average \
            else torch.cat(context, dim=2)

    def forward(self, src_seq, src_len, state=None):
        seq_mask = self.sequence_mask(src_seq.size(), src_len)
        h_gt_1 = state[0][-self.num_g:]
        h_wt_1 = state[0][:-self.num_g].masked_fill(seq_mask, 0)
        c_gt_1 = state[1][-self.num_g:]
        c_wt_1 = state[1][:-self.num_g].masked_fill(seq_mask, 0)
        h_hat = h_wt_1.mean(dim=0)
        fg = F.sigmoid(F.linear(h_gt_1, self.s_wg) +
                       F.linear(h_hat, self.s_ug) +
                       self.s_bg)
        o = F.sigmoid(F.linear(h_gt_1, self.s_wo) +
                      F.linear(h_hat, self.s_uo) + self.s_bo)
        fi = F.sigmoid(F.linear(h_gt_1, self.s_wf) +
                       F.linear(h_wt_1, self.s_uf) +
                       self.s_bf).masked_fill(seq_mask, -1e25)
        fi_normalized = F.softmax(fi, dim=0)
        c_gt = fg.mul(c_gt_1).add(fi_normalized.mul(c_wt_1).sum(dim=0))
        h_gt = o.mul(F.tanh(c_gt))
        epsilon = self.in_window_context(h_wt_1, window_size=self.n_windows)
        i = F.sigmoid(F.linear(epsilon, self.w_wi) +
                      F.linear(src_seq, self.w_ui) +
                      F.linear(h_gt_1, self.w_vi) + self.w_bi)
        l = F.sigmoid(F.linear(epsilon, self.w_wl) +
                      F.linear(src_seq, self.w_ul) +
                      F.linear(h_gt_1, self.w_vl) + self.w_bl)
        r = F.sigmoid(F.linear(epsilon, self.w_wr) +
                      F.linear(src_seq, self.w_ur) +
                      F.linear(h_gt_1, self.w_vr) + self.w_br)
        f = F.sigmoid(F.linear(epsilon, self.w_wf) +
                      F.linear(src_seq, self.w_uf) +
                      F.linear(h_gt_1, self.w_vf) + self.w_bf)
        s = F.sigmoid(F.linear(epsilon, self.w_ws) +
                      F.linear(src_seq, self.w_us) +
                      F.linear(h_gt_1, self.w_vs) + self.w_bs)
        o = F.sigmoid(F.linear(epsilon, self.w_wo) +
                      F.linear(src_seq, self.w_uo) +
                      F.linear(h_gt_1, self.w_vo) + self.w_bo)
        u = F.tanh(F.linear(epsilon, self.w_wu) +
                   F.linear(src_seq, self.w_uu) +
                   F.linear(h_gt_1, self.w_vu) + self.w_bu)
        gates = torch.stack((l, f, r, s, i), dim=0)
        gates_normalized = F.softmax(gates.masked_fill(seq_mask, -1e25), dim=0)
        c_wt_l, c_wt_1, c_wt_r = \
            self.in_window_context(c_wt_1).chunk(3, dim=2)
        c_mergered = torch.stack((c_wt_l, c_wt_1, c_wt_r,
                                  c_gt_1.expand_as(c_wt_1.data), u), dim=0)

        c_wt = gates_normalized.mul(c_mergered).sum(dim=0)
        c_wt = c_wt.masked_fill(seq_mask, 0)
        h_wt = o.mul(F.tanh(c_wt))

        h_t = torch.cat((h_wt, h_gt), dim=0)
        c_t = torch.cat((c_wt, c_gt), dim=0)
        return (h_t, c_t)


class sLSTM(nn.Module):
    def __init__(self, d_word, n_src_vocab, n_tgt_vocab, n_intent, d_hidden, window_size=3,
                 steps=5, sentence_nodes=1, bias=True,
                 batch_first=False, dropout=0.5, embeddings=None):
        super(sLSTM, self).__init__()
        self.sw1 = nn.Sequential(nn.Conv1d(d_hidden, d_hidden, kernel_size=1, padding=0),
                                 nn.BatchNorm1d(d_hidden), nn.ReLU())
        self.sw3 = nn.Sequential(nn.Conv1d(d_hidden, d_hidden, kernel_size=1, padding=0),
                                 nn.ReLU(), nn.BatchNorm1d(d_hidden),
                                 nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1),
                                 nn.ReLU(), nn.BatchNorm1d(d_hidden))
        self.sw33 = nn.Sequential(nn.Conv1d(d_hidden, d_hidden, kernel_size=1, padding=0),
                                  nn.ReLU(), nn.BatchNorm1d(d_hidden),
                                  nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding=2),
                                  nn.ReLU(), nn.BatchNorm1d(d_hidden))
        self.linear = nn.Sequential(nn.Linear(2 * d_hidden, 2 * d_hidden), nn.GLU(),
                                    nn.Dropout(dropout))
        self.multi_att = StackedSelfAttentionEncoder(input_dim=d_hidden, hidden_dim=d_hidden,
                                                     projection_dim=d_hidden,
                                                     feedforward_hidden_dim=2 * d_hidden, num_layers=2,
                                                     num_attention_heads=5)
        self.filter_linear = nn.Linear(3 * d_hidden, d_hidden)
        self.steps = steps
        self.d_hidden = d_hidden
        self.sentence_nodes = sentence_nodes
        self.n_tgt_vocab = n_tgt_vocab
        self.n_intent = n_intent
        self.elmo = Elmo(Constants.ELMO_OPTIONS, Constants.ELMO_WEIGHT, 1, requires_grad=False, dropout=dropout)
        # self.elmo = Elmo(Constants.ELMO_OPTIONS, Constants.ELMO_WEIGHT, 1, requires_grad=True, dropout=dropout) # better results can be achieved, at the cost of training time
        self.slot_out = nn.Linear(d_hidden, n_tgt_vocab)
        self.intent_out = nn.Linear(d_hidden, n_intent)
        self.dropout = nn.Dropout(dropout)
        self.cell = sLSTMCell(d_word=1024, d_hidden=d_hidden,
                              n_windows=window_size,
                              n_sent_nodes=sentence_nodes, bias=bias,
                              batch_first=batch_first)
        self.input_out = nn.Linear(d_word * 3, d_word)
        self.sigmoid = nn.Sigmoid()

    def _get_conv(self, src):
        old = src
        src = src.transpose(0, 1).transpose(1, 2)  # (l,b,d) ->(b,l,d) ->(b,d,l)
        conv1 = self.sw1(src)
        conv3 = self.sw3(src)
        conv33 = self.sw33(src)
        conv = torch.cat([conv1, conv3, conv33], dim=1)  # (b,3d,l)
        conv = self.filter_linear(conv.transpose(1, 2)).transpose(0, 1)  # (b,3d,l)->(b,l,3d)-> (b,l,d) -> (l,b,d)
        conv += old
        return conv

    def _get_self_attn(self, src, mask):
        attn = self.multi_att(src, mask)
        attn += src
        return attn

    def forward(self, src_seq, src_char, state=None):
        mask = src_seq.gt(Constants.PAD)
        src_len = torch.cuda.LongTensor(np.array(get_len(src_seq.cpu().transpose(0, 1)))).unsqueeze(1)
        elmo_embs = self.elmo(src_char)['elmo_representations'][0]  # (b,l,d)
        elmo_embs = elmo_embs.transpose(0, 1)
        src_embs = elmo_embs
        if state is None:
            h_t = torch.zeros(src_embs.size(0) + self.sentence_nodes,
                              src_embs.size(1),
                              self.d_hidden).cuda()
            c_t = torch.zeros_like(h_t)
        else:
            h_t = state[0]
            c_t = state[1]
        for step in range(self.steps):
            h_t, c_t = self.cell(src_embs, src_len, (h_t, c_t))
        h_t = self.dropout(h_t)
        h_w = h_t[:-self.sentence_nodes]  # (l,b,d)
        h_s = h_t[-self.sentence_nodes:]
        conv = self._get_conv(h_w)
        attn = self._get_self_attn(conv, mask)
        attn_gate = self.sigmoid(attn)
        h_w *= attn_gate
        slot_logit = self.slot_out(h_w).view(-1, self.n_tgt_vocab)  # (l,b,n_tgt)
        sent_logit = self.intent_out(h_s).view(-1, self.n_intent)  # (1,b,n_intent)
        return slot_logit, sent_logit
