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

    train_data, val_data = process_dataset(dataset, test_set_size)
    transform_data(train_data, list(range(len(train_data.X))))
    val_idx = random.sample(range(len(val_data.X)), min(test_set_size, len(val_data.X)))
    transform_data(val_data, val_idx)
    print('loaded data', dataset)
    print('num train samples:', len(train_data.X))
    print('num val samples:', len(val_data.X))
    regs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 
            5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    best_model = None
    best_acc = float('-inf')
    for reg in regs:
        lr = LogisticRegression(max_iter=1000, C=reg)
        lr.fit(np.stack(train_data.X, axis=0), np.array(train_data.Y))
        train_predictions = lr.predict(np.stack(train_data.X, axis=0))
        train_accuracy = np.mean(train_predictions == np.array(train_data.Y))
        val_predictions = lr.predict(np.stack(val_data.X, axis=0))
        val_accuracy = np.mean(val_predictions == np.array(val_data.Y))
        print(f'regularization: {reg}, val accuracy: {val_accuracy}')
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model = lr
    beta_logistic = best_model.coef_
    if not os.path.exists('outputs/'):
        os.makedirs('outputs/')
    f_path = os.path.join('outputs', '{dataset_name}_logistic_beta.pkl'.format(dataset_name=dataset))
    pkl.dump(beta_logistic, open(f_path, 'wb'))
    print('best accuracy model %.3f saved' % (best_acc))



