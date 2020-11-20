import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import json

datasets = ['IMDB', 'Furd', 'AG_News', 'Newsgroups', 'Amzn', 'Yelp', 'SMS']
embed_key = 'embed_beta'
metrics = [TopPercentMatch(p=5), KendallTauCorr()]
split_perf_groups = [('train', 'train_acc'), ('val', 'test_acc')]
runs = ['normal_B', 'normal_C', 'normal_D', 'normal_E', 'normal_F']

all_results = {}
for dataset in datasets:
	dat = load_dataset_dict(dataset, embed_key)
	for metric in metrics:
		for split, perf in split_perf_groups:
			stats = fetch_stats(dat, split, metric, 'normal_A', runs, 'uniform', embed_key, perf)
			all_results[(dataset, metric.name, split, perf)] = stats
			print('%s, %s, %s, %s' % (dataset, metric.name, split, perf))
			print(stats)

with open('classification_results.pkl', 'wb+') as f:
	pkl.dump(all_results, f)