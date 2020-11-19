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
from eval_model import evaluate, evaluate_next_token, get_state_scores, get_state_scores2, get_grad_influence2

def dump_interpret(model_path, full_model, invasive_uniform, eval_bleu, dataset, include_train_subset, grad_bsize):
    print('interpreting %s' % model_path)
    meta_stats = {}

    training_data, validation_data, vocab = load_dataset_by_name(dataset)

    pad_id = vocab.PieceToId("<pad>")
    bos_id = vocab.PieceToId("<s>")
    eos_id = vocab.PieceToId("</s>")

    val_data_manager = StateManager(validation_data, vocab, bos_id, eos_id, pad_id, device, model_config)
    train_data_manager = StateManager(training_data, vocab, bos_id, eos_id, pad_id, device, model_config)
    VOCAB_SIZE = vocab.GetPieceSize()

    model = Seq2seq(device=device, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, dropout=0,
                    attn_lambda=0.0, pad_id=pad_id, full_model=full_model, invasive_uniform=invasive_uniform).to(device)
    model.load_state_dict(torch.load(model_path))

    if not full_model:
        state_scores_val = get_state_scores(model, val_data_manager)
    else:
        state_scores_val = get_state_scores2(model, val_data_manager)
    grad_influence_val = get_grad_influence2(model, val_data_manager, grad_bsize)

    perplexity_val, acc_val, attn_val = evaluate_next_token(model, val_data_manager)
    meta_stats['val_acc'] = acc_val
    meta_stats['val_perplexity'] = perplexity_val

    if eval_bleu:
      bleu_val = evaluate(model, val_data_manager, method='beam')
      meta_stats['val_bleu'] = bleu_val

    if include_train_subset:
      random.seed(1)
      train_idxs = random.sample(range(len(train_data_manager.dataset)), k=len(val_data_manager.dataset))
      inverse_train_idx_map = {train_idxs[i]: i for i in range(len(train_idxs))}
      eval_train = StateManager([train_data_manager.dataset[idx] for idx in train_idxs], vocab, bos_id, eos_id, pad_id, device, model_config)
      if not full_model:
          state_scores_train = get_state_scores(model, eval_train)
      else:
          state_scores_train = get_state_scores2(model, eval_train)
      grad_influence_train = get_grad_influence2(model, eval_train, grad_bsize)

      perplexity_train, acc_train, attn_train = evaluate_next_token(model, eval_train)
      meta_stats['train_acc'] = acc_train
      meta_stats['train_perplexity'] = perplexity_train

      if eval_bleu:
        bleu_train = evaluate(model, eval_train, method='beam')
        meta_stats['train_bleu'] = bleu_train

    items = []
    for i in range(len(val_data_manager.dataset)):
        curr_dict = {}
        curr_dict['split'] = 'val'
        curr_dict['src'] = sentence2ids_nopad(val_data_manager, val_data_manager.dataset[i].src, additional_eos=False)
        curr_dict['trg'] = sentence2ids_nopad(val_data_manager, val_data_manager.dataset[i].trg, additional_eos=False)
        curr_dict['beta'] = state_scores_val[i]
        curr_dict['alpha'] = attn_val[i]
        curr_dict['grad'] = grad_influence_val[i]

        items.append(curr_dict)

    if include_train_subset:
      train_idxs_set = set(train_idxs)
      for i in range(len(train_data_manager.dataset)):
        curr_dict = {}
        curr_dict['split'] = 'train'
        curr_dict['src'] = sentence2ids_nopad(train_data_manager, train_data_manager.dataset[i].src, additional_eos=False)
        curr_dict['trg'] = sentence2ids_nopad(train_data_manager, train_data_manager.dataset[i].trg, additional_eos=False)
        if i in train_idxs_set:
          curr_dict['beta'] = state_scores_train[inverse_train_idx_map[i]]
          curr_dict['alpha'] = attn_train[inverse_train_idx_map[i]]
          curr_dict['grad'] = grad_influence_train[inverse_train_idx_map[i]]
        else:
          curr_dict['beta'] = None
          curr_dict['alpha'] = None
          curr_dict['grad'] = None

        items.append(curr_dict)

    return items, meta_stats

def merge_dicts(dicts):
  master = []
  for key_ in dicts:
    for i, item in enumerate(dicts[key_]):
      if i >= len(master):
        master.append({'src': item['src'], 'trg': item['trg'], 'split': item['split']})
      master[i][tuple(list(key_) + ['alpha'])] = item['alpha']
      master[i][tuple(list(key_) + ['beta'])] = item['beta']
      master[i][tuple(list(key_) + ['grad'])] = item['grad']
  return master



def get_args():
  parser = ArgumentParser()
  parser.add_argument('--config', type=str, default='configs/model.json')
  parser.add_argument('--dataset', type=str)
  parser.add_argument('--eval_bleu', default=False, action='store_true')
  parser.add_argument('--include_train_subset', default=False, action='store_true')
  parser.add_argument('--grad_bsize', type=int, default=16)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_args()

  assert torch.cuda.is_available()
  # device = torch.device("cpu")
  device = torch.device("cuda")
  print("Using device:", device)

  with open(args.config, 'r') as f:
    model_config = json.load(f)

  NUM_LAYERS = model_config['num_layers']
  DROPOUT = model_config['dropout']
  HIDDEN_DIM = model_config['hidden_dim']
  batch_size = model_config['batch_size']
  include_train_subset = args.include_train_subset
  grad_bsize = args.grad_bsize

  eval_list = 'configs/%s_log.json' % (args.dataset)
  with open(eval_list, 'r') as f:
    evals = json.load(f)

  dicts = {}
  metas = {}
  for eval_ in evals:
    if eval_['uniform']:
      config_name = 'h_dim=%d,dropout=%f,b_size=%d,seed=%d,uniform' % (HIDDEN_DIM, DROPOUT, batch_size, eval_['seed'])
    else:
      config_name = 'h_dim=%d,dropout=%f,b_size=%d,seed=%d,normal' % (HIDDEN_DIM, DROPOUT, batch_size, eval_['seed'])
    model_path = ("models/%s/%s/" % (args.dataset, config_name))
    assert os.path.exists(os.path.join(model_path, 'model' + str(eval_['iteration'])))
    
    items, meta_stats = dump_interpret(os.path.join(model_path, 'model' + str(eval_['iteration'])), 
                      full_model=True, invasive_uniform=eval_['uniform'], eval_bleu=args.eval_bleu, dataset=args.dataset,
                      include_train_subset=include_train_subset, grad_bsize=grad_bsize)
    
    dicts[(eval_['name'], eval_['iteration'])] = items
    for key_ in meta_stats:
      metas[(eval_['name'], eval_['iteration'], key_)] = meta_stats[key_]

  merged_dicts = merge_dicts(dicts)
  combined = {'metas': metas, 'data': merged_dicts}
  if not os.path.exists('outputs/'):
    os.makedirs('outputs/')
  with open('outputs/%s_logs.pkl' % (args.dataset), 'wb+') as f:
    pkl.dump(combined, f)



