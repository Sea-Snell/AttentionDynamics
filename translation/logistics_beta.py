import pickle as pkl
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm

dataset = 'multi30k'
for dataset_name in ['multi30k', 'iwslt14']:
    dataset = pkl.load(open('data/{dataset_name}_logs_rz.pkl'.format(dataset_name=dataset_name), 'rb'))
    train_set = [d for d in dataset if d['split'] == 'train']
    data, row, col = [], [], []
    Y, c = [], 0
    for d in tqdm(train_set):
        src, trg = d['src'], d['trg']

        for t in trg[1:]:
            for s in src:
                data.append(1. / len(src))
                row.append(c)
                col.append(s)
            Y.append(t)
            c += 1
    dim1, dim2 = c, np.max(col) + 1

    X = csr_matrix((data, (row, col)), shape=(dim1, dim2))
    Y = np.array(Y)
    model = LogisticRegression()
    model.fit(X, Y)
    coef = model.coef_.T
    coef = np.array(coef)
    np.savetxt('data/{dataset_name}_logistics_coef'.format(dataset_name=dataset_name), coef)
