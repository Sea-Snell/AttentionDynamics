import pickle as pkl
from argparse import ArgumentParser
from tqdm import tqdm
import random
import os
from load_datasets import process_dataset
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_set_size', type=int, default=4000)
    parser.add_argument('--train_set_size', type=int, default=25000)
    args = parser.parse_args()
    return args

def transform_data(dat, idxs):
    new_xs = []
    new_ys = []
    for i in idxs:
        new_x = np.zeros((dat.vocab_size))
        for idx in dat.X[i]:
            new_x[idx] += 1
        new_x /= len(dat.X[i])
        new_xs.append(new_x)
        new_ys.append(dat.Y[i])
    dat.X = new_xs
    dat.Y = new_ys

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    test_set_size = args.test_set_size
    train_set_size = args.train_set_size

    train_data, val_data = process_dataset(dataset, test_set_size)
    print(len(train_data.X))
    train_idx = random.sample(range(len(train_data.X)), min(train_set_size, len(train_data.X)))
    transform_data(train_data, train_idx)
    val_idx = random.sample(range(len(val_data.X)), min(test_set_size, len(val_data.X)))
    transform_data(val_data, val_idx)
    print('loaded data')
    print('num train samples:', len(train_data.X))
    print('num val samples:', len(val_data.X))
    lr = LogisticRegression(max_iter=1000)
    print('fitting')
    lr.fit(np.stack(train_data.X, axis=0), np.array(train_data.Y))
    print('predicting')
    train_predictions = lr.predict(np.stack(train_data.X, axis=0))
    train_accuracy = np.mean(train_predictions == np.array(train_data.Y))
    print('train accuracy', train_accuracy)
    val_predictions = lr.predict(np.stack(val_data.X, axis=0))
    val_accuracy = np.mean(val_predictions == np.array(val_data.Y))
    print('val accuracy', val_accuracy)



