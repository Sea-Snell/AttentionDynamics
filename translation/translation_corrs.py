import pickle as pkl
import numpy as np
from utils import *

def max_acc_iter(name, metas):
    max_acc = -1.0
    max_acc_iter = None
    for key_ in metas:
        if key_[0] == name and key_[2] == 'val_bleu':
            if metas[key_] > max_acc:
                max_acc = metas[key_]
                max_acc_iter = key_[1]
    return max_acc_iter, max_acc

dataset = 'news_commentary_v14_en_nl'
dat = pkl.load(open('outputs/{dataset}_logs.pkl'.format(dataset=dataset), 'rb'))
all_dicts = dat['data']
subset = 'val'
dicts = [d for d in all_dicts if d['split'] == 'val']
iterations = sorted(list(set([key[1] for key in dat['metas']])))

metrics = [TopPercentMatch(p=5)]
normalA_iter, normal_A_bleu = max_acc_iter('normal_A', dat['metas'])
normalB_iter, normal_B_bleu = max_acc_iter('normal_B', dat['metas'])
uniform_iter, uniform_bleu = max_acc_iter('uniform', dat['metas'])



