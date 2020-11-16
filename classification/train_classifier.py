from torch import nn
import json
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle as pkl
from argparse import ArgumentParser
import random
import numpy as np
import tqdm
from classifier_model import Model
from argparse import ArgumentParser
import os
from load_datasets import process_dataset, DataManager

def train(model, model_path, train_data, test_data, steps, bsize, save_every, device, uniform):
    model.train()

    shuffle_every = len(train_data.X) // bsize
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for step in tqdm.trange(steps):
        if step % shuffle_every == 0:
            ordering = [_ for _ in range(len(train_data.X))]
            random.shuffle(ordering)
            train_data.X = [train_data.X[i] for i in ordering]
            train_data.Y = [train_data.Y[i] for i in ordering]

        if step % save_every == 0:
            val_results = defaultdict(list)
            test_results = defaultdict(list)

            model.eval()
            for i in range(0, len(test_data.X), bsize):
                test_result_dict = model(test_data.X[i:(i+bsize)], uniform=uniform, in_grad=False, pad_token=test_data.stoi['<pad>'])
                test_results['gold'].extend([1 if (t > 0.5) else 0 for t in test_data.Y[i:(i+bsize)]])
                test_results['predicted'].extend([1 if (s > 0) else 0 for s in test_result_dict['scores'].detach().cpu().numpy()])

            test_acc = np.mean([1 if a == b else 0 for a, b in zip(test_results['gold'], test_results['predicted'])])
            print('accuracy on test set:', test_acc)

            torch.save(model.state_dict(), model_path + str(step))

        model.train()
        optimizer.zero_grad()
        input_X, Y = [[l[i % len(train_data.X)] for i in range(step * bsize, (step + 1) * bsize)] for l in [train_data.X, train_data.Y]]
        output_dict = model(input_X, uniform=uniform, in_grad=False, pad_token=train_data.stoi['<pad>'])
        scores = output_dict['scores']
        loss = loss_func(scores, torch.tensor(Y, dtype=torch.float32).to(device))
        loss.backward()
        optimizer.step()

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--uniform', default=False, action='store_true')
    parser.add_argument('--config', type=str, default='configs/model.json')
    parser.add_argument('--test_set_size', type=int, default=4000)
    parser.add_argument('--save_every', type=int, default=250)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    seed = args.seed
    dataset = args.dataset
    steps = args.steps
    uniform = args.uniform
    save_every = args.save_every

    torch.manual_seed(seed)
    random.seed(seed)

    with open(args.config, 'r') as f:
        model_config = json.load(f)

    HIDDEN_DIM = model_config['hidden_dim']
    INTERMEDIATE_DIM = model_config['intermediate_dim']
    EMBED_DIM = model_config['embed_dim']
    bsize = model_config['batch_size']

    if uniform:
        config_name = 'h_dim=%d,b_size=%d,seed=%d,uniform' % (HIDDEN_DIM, bsize, seed)
    else:
        config_name = 'h_dim=%d,b_size=%d,seed=%d,normal' % (HIDDEN_DIM, bsize, seed)

    model_path = ("models/%s/%s/" % (dataset, config_name))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_data_manager, test_data_manager = process_dataset(dataset, args.test_set_size)
    model = Model(train_data_manager.vocab_size, train_data_manager.tokenid2vector, EMBED_DIM, HIDDEN_DIM, INTERMEDIATE_DIM, device).to(device)
    train(model, os.path.join(model_path, 'model'), train_data_manager, test_data_manager, steps, bsize, save_every, device, uniform)














