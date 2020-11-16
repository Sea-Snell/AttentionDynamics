import pickle as pkl
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm
import os
from load_datasets import process_dataset, DataManager
from argparse import ArgumentParser
from collections import defaultdict

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset

    train_data, _ = process_dataset(dataset, 0)
    
    data, row, col = [], [], []
    for i in tqdm(range(len(train_data.X))):
        x, y = train_data.X[i], train_data.Y[i]
        for tok in x:
            data.append(1. / len(x))
            row.append(i)
            col.append(tok)
    dim1, dim2 = len(train_data.X), np.max(col) + 1

    X = csr_matrix((data, (row, col)), shape=(dim1, dim2))
    Y = np.array(train_data.Y)
    model = LogisticRegression()
    model.fit(X, Y)
    coef = model.coef_.T
    coef = np.array(coef)
    if not os.path.exists('outputs/'):
        os.makedirs('outputs/')
    with open('outputs/{dataset_name}_logistics_beta.pkl'.format(dataset_name=dataset), 'wb+') as f:
        pkl.dump(coef, f)



