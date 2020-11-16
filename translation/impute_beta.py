import pickle as pkl
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--grid_type', choices=['embedding', 'IBM'])
parser.add_argument('--dataset', type=str)
parser.add_argument('--embed_dim', type=int, default=None, help="if using embedding transltion matrix, the embedding dim used for training")
parser.add_argument('--impute_train', default=False, action='store_true')
args = parser.parse_args()

if args.grid_type == 'embedding':
	assert args.embed_dim is not None, 'embdding dim should be specified with grid_type=embedding'

dataset_name = args.dataset
dict_f_path = 'outputs/{dataset_name}_logs.pkl'.format(dataset_name=dataset_name)
dat = pkl.load(open(dict_f_path, 'rb'))
dicts = dat['data']

if args.grid_type == 'embedding':
	embed_beta = pkl.load(open('outputs/{dataset_name}embedding{embed_dim}translation.pkl'.format(embed_dim=args.embed_dim, dataset_name=dataset_name), 'rb'))
	key_name = 'embed{embed_dim}_beta'.format(embed_dim=args.embed_dim)
elif args.grid_type == 'IBM':
	embed_beta = pkl.load(open('outputs/IBM_translation_{dataset_name}.pkl'.format(dataset_name=dataset_name), 'rb'))
	key_name = 'IBM_beta'

for d in dicts:
	if not args.impute_train and d['split'] == 'train':
		continue
	src, trg = d['src'], d['trg']
	betas = []
	for t in trg[1:]:
	    beta = []
	    for s in src:
	        beta.append(embed_beta[s][t])
	    betas.append(beta)
	betas = np.array(betas)
	d[key_name] = betas

pkl.dump({'data': dicts, 'metas': dat['metas']}, open(dict_f_path, 'wb'))
