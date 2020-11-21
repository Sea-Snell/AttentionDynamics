import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import json

datasets = ['multi30k']
embed_key = 'embed_beta'
# metrics = [TopPercentMatch(p=5), KendallTauCorr()]
# split_perf_groups = [('train', 'train_acc'), ('train', 'train_bleu'), ('val', 'val_acc'), ('val', 'val_bleu')]
metrics = [TopPercentMatch(p=5)]
split_perf_groups = [('val', 'val_acc'), ('val', 'val_acc')]
runs = ['normal_B']

all_results = {}
for dataset in datasets:
	dat = load_dataset_dict(dataset, embed_key)
	for metric in metrics:
		for split, perf in split_perf_groups:
			stats = fetch_stats(dat, split, metric, 'normal_A', runs, 'uniform', embed_key, perf)
			all_results[(dataset, metric.name, split, perf)] = stats
			print('%s, %s, %s, %s' % (dataset, metric.name, split, perf))
			print(stats)

# with open('translation_results.pkl', 'wb+') as f:
# 	pkl.dump(all_results, f)