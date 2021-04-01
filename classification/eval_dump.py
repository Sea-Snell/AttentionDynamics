import json
import torch
import pickle as pkl
import random
import numpy as np
from classifier_model import Model
from argparse import ArgumentParser
import os
from load_datasets import process_dataset, DataManager


def dump_interpret(model_path, uniform, dataset, test_set_size, model_config, device):
    print('interpreting %s' % model_path)
    train_data, test_data = process_dataset(dataset, test_set_size)

    model = Model(train_data.vocab_size, train_data.tokenid2vector, 
                  model_config['embed_dim'], model_config['hidden_dim'], 
                  model_config['intermediate_dim'], device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.train()

    bsize = model_config['batch_size']

    data_stats = []
    for i in range(0, len(test_data.X), bsize):
      result_dict = model(test_data.X[i:(i+bsize)], uniform=uniform, in_grad=True, pad_token=test_data.stoi['<pad>'])
      scores = result_dict['scores'].detach().cpu().numpy()
      outputs = -(torch.tensor(list(test_data.Y[i:(i+bsize)])).to(device) * torch.log(torch.sigmoid(result_dict['scores'])) + (1 - torch.tensor(list(test_data.Y[i:(i+bsize)])).to(device)) * torch.log(1 - torch.sigmoid(result_dict['scores'])))
      gs = [-g.detach().cpu().numpy()[:result_dict['lens'][i]]  for i, g in enumerate(model.influence(outputs))]
      for x in range(len(result_dict['alpha'])):
        item = {}
        item['grad'] = gs[x]
        item['alpha'] = result_dict['alpha'][x]
        item['beta'] = result_dict['beta'][x]
        item['Y'] = int(test_data.Y[i+x] > 0.5)
        item['predicted'] = int(scores[x] > 0)
        item['X'] = test_data.X[i+x]
        item['split'] = 'val'
        data_stats.append(item)

    random.seed(1)
    train_idxs = random.sample(range(len(train_data.X)), k=len(test_data.X))
    inverse_train_idx_map = {train_idxs[i]: i for i in range(len(train_idxs))}
    eval_train = DataManager([train_data.X[idx] for idx in train_idxs], [train_data.Y[idx] for idx in train_idxs], train_data.tokenid2vector, train_data.vocab_size, train_data.stoi)
    train_idxs_set = set(train_idxs)

    temp_train_stats = []
    for i in range(0, len(eval_train.X), bsize):
      result_dict = model(eval_train.X[i:(i+bsize)], uniform=uniform, in_grad=True, pad_token=train_data.stoi['<pad>'])
      scores = result_dict['scores'].detach().cpu().numpy()
      outputs = -(torch.tensor(list(eval_train.Y[i:(i+bsize)])).to(device) * torch.log(torch.sigmoid(result_dict['scores'])) + (1 - torch.tensor(list(eval_train.Y[i:(i+bsize)])).to(device)) * torch.log(1 - torch.sigmoid(result_dict['scores'])))
      gs = [-g.detach().cpu().numpy()[:result_dict['lens'][i]]  for i, g in enumerate(model.influence(outputs))]
      for x in range(len(result_dict['alpha'])):
        item = {}
        item['grad'] = gs[x]
        item['alpha'] = result_dict['alpha'][x]
        item['beta'] = result_dict['beta'][x]
        item['Y'] = int(eval_train.Y[i+x] > 0.5)
        item['predicted'] = int(scores[x] > 0)
        item['X'] = eval_train.X[i+x]
        item['split'] = 'train'
        temp_train_stats.append(item)

    for i in range(len(train_data.X)):
      if i in train_idxs_set:
        data_stats.append(temp_train_stats[inverse_train_idx_map[i]])
      else:
        data_stats.append({'alpha': None, 'beta': None, 'predicted': None, 'X': train_data.X[i], 'Y': train_data.Y[i], 'split': 'train', 'grad': None})

    test_acc = np.mean([1 if item['predicted'] == item['Y'] else 0 for item in data_stats if item['split'] == 'val'])
    train_acc = np.mean([1 if item['predicted'] == item['Y'] else 0 for item in data_stats if item['split'] == 'train' and item['predicted'] is not None])
    meta_stats = {'train_acc': train_acc, 'test_acc': test_acc}

    return data_stats, meta_stats

def merge_dicts(dicts):
  master = []
  for key_ in dicts:
    for i, item in enumerate(dicts[key_]):
      if i >= len(master):
        master.append({'src': item['X'], 'trg': item['Y'], 'split': item['split']})
      master[i][tuple(list(key_) + ['alpha'])] = item['alpha']
      master[i][tuple(list(key_) + ['beta'])] = item['beta']
      master[i][tuple(list(key_) + ['grad'])] = item['grad']
  return master



def get_args():
  parser = ArgumentParser()
  parser.add_argument('--config', type=str, default='configs/model.json')
  parser.add_argument('--dataset', type=str)
  parser.add_argument('--test_set_size', type=int, default=4000)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_args()
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  print("Using device:", device)

  with open(args.config, 'r') as f:
    model_config = json.load(f)

  HIDDEN_DIM = model_config['hidden_dim']
  INTERMEDIATE_DIM = model_config['intermediate_dim']
  EMBED_DIM = model_config['embed_dim']
  bsize = model_config['batch_size']

  eval_list = 'configs/%s_log.json' % (args.dataset)
  with open(eval_list, 'r') as f:
    evals = json.load(f)
  dicts = {}
  metas = {}
  for eval_ in evals:
    if eval_['uniform']:
        config_name = 'h_dim=%d,b_size=%d,seed=%d,uniform' % (HIDDEN_DIM, bsize, eval_['seed'])
    else:
        config_name = 'h_dim=%d,b_size=%d,seed=%d,normal' % (HIDDEN_DIM, bsize, eval_['seed'])
    model_path = ("models/%s/%s/" % (args.dataset, config_name))
    assert os.path.exists(os.path.join(model_path, 'model' + str(eval_['iteration'])))

    items, meta_stats = dump_interpret(os.path.join(model_path, 'model' + str(eval_['iteration'])), 
                                      eval_['uniform'], args.dataset, args.test_set_size, model_config, device)
    
    dicts[(eval_['name'], eval_['iteration'])] = items
    for key_ in meta_stats:
      metas[(eval_['name'], eval_['iteration'], key_)] = meta_stats[key_]

  merged_dicts = merge_dicts(dicts)
  combined = {'metas': metas, 'data': merged_dicts}
  if not os.path.exists('outputs/'):
    os.makedirs('outputs/')
  with open('outputs/%s_logs.pkl' % (args.dataset), 'wb+') as f:
    pkl.dump(combined, f)



