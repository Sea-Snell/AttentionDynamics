import torch
import torchtext

import json
import math
import random

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
import sentencepiece
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from seq2seq_model import Seq2seq
from argparse import ArgumentParser
import numpy as np
import pickle as pkl
import os
from load_datasets import load_dataset_by_name, StateManager, sentence2ids_nopad
device = torch.device('cpu')

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset_name = args.dataset

training_data, validation_data, vocab = load_dataset_by_name(dataset_name)
pad_id = vocab.PieceToId("<pad>")
bos_id = vocab.PieceToId("<s>")
eos_id = vocab.PieceToId("</s>")
val_data_manager = StateManager(validation_data, vocab, bos_id, eos_id, pad_id, device, {})
train_data_manager = StateManager(training_data, vocab, bos_id, eos_id, pad_id, device, {})

co_occurance_grid = np.zeros(shape=(len(vocab), len(vocab)))

for item in train_data_manager.dataset:
  src_ = sentence2ids_nopad(train_data_manager, item.src, additional_eos=False)
  trg_ = sentence2ids_nopad(train_data_manager, item.trg, additional_eos=False)[1:]
  for in_tok in src_:
    for out_tok in trg_:
      co_occurance_grid[in_tok, out_tok] += 1/len(src_)

if not os.path.exists('outputs/'):
    os.makedirs('outputs/')
with open(os.path.join('outputs', 'IBM_translation_%s.pkl' % (dataset_name)), 'wb+') as f:
  pkl.dump(co_occurance_grid, f)








