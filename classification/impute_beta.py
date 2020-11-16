import pickle as pkl
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--impute_train', default=False, action='store_true')
args = parser.parse_args()

dataset_name = args.dataset
dict_f_path = 'outputs/{dataset_name}_logs.pkl'.format(dataset_name=dataset_name)
dat = pkl.load(open(dict_f_path, 'rb'))
dicts = dat['data']

logistics_beta = pkl.load(open('outputs/{dataset_name}_logistics_beta.pkl'.format(dataset_name=dataset_name), 'rb'))
key_name = 'logistics_beta'

for d in dicts:
	if not args.impute_train and d['split'] == 'train':
		continue
	src = d['src']
	beta = []
	for s in src:
	    beta.append(logistics_beta[s])
	d[key_name] = beta

pkl.dump({'data': dicts, 'metas': dat['metas']}, open(dict_f_path, 'wb'))
