import torch
from torch import nn
import os
import pickle as pkl
import numpy as np
from typing import List
import random
from itertools import combinations
from collections import Counter

exp_name = 're_all_comb112'
exp_dir = '../logs/seq2seq_toylogs2/' + exp_name
save_result_dir = exp_dir + '/results/'
num_vocab = 40
end_of_sequence = num_vocab
start_of_sequence = num_vocab + 1
seq_length = 40
device = 'cuda' if torch.cuda.is_available() else 'cpu'
experiment_num_heads = 20
NUM_TRIALS = 5
num_samples = 50
num_heads = 8


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


metrics = {}
def register_metrics(metric_name):
    def decorator(f):
        metrics[metric_name] = f
    return decorator


@register_metrics('acc')
def acc(return_dict, X, Y):
    final_probs = return_dict['probs']
    ans = np.argmax(final_probs, axis=-1)
    return np.mean(np.equal(ans, Y[:, 1:].cpu().numpy()))


@register_metrics('attn_acc')
def attn_acc(return_dict, X, Y):
    weights = return_dict['attn_weights']
    attn_accs = []
    for head_attn_weight in weights:
        max_attn = np.argmax(head_attn_weight, axis=-1)
        gold_attn = np.array([[i for i in range(seq_length)] for _ in range(X.size()[0])])
        attn_accs.append(np.mean(np.equal(max_attn[:, :-1], gold_attn)))
    return attn_accs


@register_metrics('attn_acc_m1')
def attn_acc_m1(return_dict, X, Y):
    weights = return_dict['attn_weights']
    attn_accs = []
    for head_attn_weight in weights:
        max_attn = np.argmax(head_attn_weight, axis=-1)
        gold_attn = np.array([[i + 1 for i in range(seq_length)] for _ in range(X.size()[0])])
        attn_accs.append(np.mean(np.equal(max_attn[:, :-1], gold_attn)))
    return attn_accs


def numpify_dict(d):
    result = {}
    result['logits'] = d['logits'].cpu().detach().numpy()
    result['probs'] = d['probs'].cpu().detach().numpy()
    result['attn_weights'] = [x.cpu().detach().numpy() for x in d['attn_weights']]
    return result


def eval_acc(m, val_set):
    m.eval()
    X, Y = val_set
    X, Y = torch.tensor(X).to(device), torch.tensor(Y).to(device)
    return_dict = numpify_dict(m(X, Y[:, :-1]))
    eval_dict = {}

    for key in metrics:
        eval_dict[key] = metrics[key](return_dict, X, Y)
    m.train()
    return eval_dict


def generate_seq(num_samples):
    X, Y = [], []
    for _ in range(num_samples):
        arr = [_ for _ in range(num_vocab)]
        random.shuffle(arr)
        x = np.array(arr[:seq_length])
        x = np.concatenate((x, [end_of_sequence]))
        y = np.concatenate(([start_of_sequence], x))
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
if not os.path.exists(save_result_dir):
    os.mkdir(save_result_dir)


class Attn_head(nn.Module):

    def __init__(self, save_path=None):
        super(Attn_head, self).__init__()
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.enc_proj = nn.Linear(hidden_dim, hidden_dim)
        if save_path is not None:
            self.load_state_dict(torch.load(save_path))


class Seq2Seq(nn.Module):


    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.enc_embedding, self.dec_embedding = [nn.Embedding(num_vocab + 2, hidden_dim) for _ in range(2)]
        self.encoder_rnn, self.decoder_rnn = [nn.GRU(hidden_dim, hidden_dim) for _ in range(2)]
        self.out_lin = nn.Linear(hidden_dim, num_vocab + 2)
        self.sm = nn.Softmax(dim=-1)
        self.after_decoder_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.Tanh()

    def set_attn(self, attn_heads: List[Attn_head]):
        self.attn_heads = nn.ModuleList(attn_heads)
        self.num_heads = len(attn_heads)

    def forward(self, X, Y_in):
        X, Y_in = X.T, Y_in.T
        enc_emb, dec_emb = self.enc_embedding(X), self.dec_embedding(Y_in)
        encoder_hidden, last = self.encoder_rnn(enc_emb)
        decoder_hidden, _ = self.decoder_rnn(dec_emb, last)

        attn_feats = [attn_head.attn_proj(decoder_hidden) for attn_head in self.attn_heads]
        attn_distrs = [self.sm(torch.einsum("tbf,sbf->bts", [attn_feat, encoder_hidden])) for attn_feat in attn_feats]
        context_vs = [torch.einsum("bts,sbf->tbf", [attn_distr, encoder_hidden]) for attn_distr in attn_distrs]
        all_context = torch.cat([attn_head.enc_proj(context_v).unsqueeze(0) for context_v, attn_head in zip(context_vs, self.attn_heads)], axis=0)
        context_v_agg = torch.mean(all_context, axis=0)

        last_hidden = self.act(self.after_decoder_hidden(torch.cat((context_v_agg, decoder_hidden), dim=-1)))
        logits = self.out_lin(last_hidden).transpose(1, 0)
        return {
            "logits": logits,
            "probs": self.sm(logits),
            "attn_weights": attn_distrs
        }


bsize = 128
hidden_dim = 256
num_steps = 200
eval_every = 10
val_set = generate_seq(400)
model_path = exp_dir + '/model.pt'


def can_learn(m, seed, return_history=False):
    # optimizer
    optim = torch.optim.Adam(m.parameters())
    loss_func = nn.CrossEntropyLoss()

    random.seed(seed)
    history = []
    for step in range(num_steps):
        if step % eval_every == 0:
            eval_dict = eval_acc(m, val_set)
            dat = {k: v for k, v in eval_dict.items()}
            dat['step'] = step
            last_acc = dat['acc']
            print(dat)
            history.append(dat)
            if last_acc > 0.95 and not return_history:
                return "good"
        X, Y = generate_seq(bsize)
        X, Y = torch.tensor(X).to(device), torch.tensor(Y).to(device)
        return_dict = m(X, Y[:, :-1])
        loss = loss_func(return_dict['logits'].permute(0, 2, 1), Y[:, 1:])
        loss.backward()
        optim.step()
        optim.zero_grad()
    if not return_history:
        if last_acc < 0.6:
            return "bad"
        return None
    return history


def get_fresh_model(attn_list):
    model = Seq2Seq()
    model.load_state_dict(torch.load(model_path))
    model.set_attn([Attn_head(attn) for attn in attn_list])
    model = model.to(device)
    return model


def is_good_single_head(attn):
    status = None
    for seed in range(3):
        result = can_learn(get_fresh_model([attn]), seed)
        if result is None:
            return None
        if status is None:
            status = result
        else:
            if status != result:
                return None
    return status


def is_good_comb(attn_list, return_history):
    return can_learn(get_fresh_model(attn_list), 0, return_history=return_history)


def init():
    model = Seq2Seq()
    torch.save(model.state_dict(), model_path)
    good_count, bad_count = 0, 0
    while bad_count < experiment_num_heads or good_count < experiment_num_heads:
        attn = Attn_head()
        cur_path = 'cur.pt'
        torch.save(attn.state_dict(), cur_path)
        head_init_property = is_good_single_head(cur_path)
        if head_init_property == 'good':
            print('A good head found')
            if good_count < experiment_num_heads:
                os.rename(cur_path, exp_dir + '/goodattn%d.pt' % good_count)
                good_count += 1
        elif head_init_property == 'bad':
            print('A bad head found')
            if bad_count < experiment_num_heads:
                os.rename(cur_path, exp_dir + '/badattn%d.pt' % bad_count)
                bad_count += 1
        if os.path.exists(cur_path):
            os.unlink(cur_path)

    print('init finished')


def run():
    random_samples = 30
    random_attns = []
    for random_idx in range(random_samples):
        attn = Attn_head()
        path = exp_dir + '/randomattn%d.pt' % random_idx
        random_attns.append(path)
        torch.save(attn.state_dict(), path)

    bad_attns = [exp_dir + '/badattn%d.pt' % idx for idx in range(experiment_num_heads)]
    good_attns = [exp_dir + '/goodattn%d.pt' % idx for idx in range(experiment_num_heads)]

    good_histories, bad_histories, random_histories = {}, {}, {}
    for idx in range(num_samples):
        random.seed(3 * idx)
        random.shuffle(bad_attns)

        comb = tuple(bad_attns[:num_heads])
        bad_histories[comb] = is_good_comb(comb, return_history=True)

        random.seed(3 * idx + 1)
        random.shuffle(bad_attns)
        random.shuffle(good_attns)
        comb = tuple(bad_attns[:num_heads - 1] + good_attns[:1])
        good_histories[comb] = is_good_comb(comb, return_history=True)

        random.seed(3 * idx + 2)
        random.shuffle(random_attns)
        comb = tuple(random_attns[:num_heads])
        random_histories[comb] = is_good_comb(comb, return_history=True)

    pkl.dump((bad_histories, good_histories, random_histories), open('re_%dheadcomb112.pkl' % num_heads, 'wb'))


def all_bad(key):
    for k in key:
        if 'bad' not in k:
            return False
    return True


if __name__ == '__main__':
    init()
    run()

