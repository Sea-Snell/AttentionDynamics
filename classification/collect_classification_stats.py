from utils import *
from argparse import ArgumentParser

#this script loads output files for a given list of datasets and produces a pickle file with all the summary statistics we present in our paper

parser = ArgumentParser()
parser.add_argument('--datasets', nargs='+', required=True, help='space seperated list of dataset names to collect stats for')
parser.add_argument('--out_file', type=str, required=True, help='file to output stats to')
parser.add_argument('--include_kendall_tau', default=False, action='store_true')
parser.add_argument('--include_train_results', default=False, action='store_true')
args = parser.parse_args()

datasets = args.datasets
out_file = args.out_file

if args.include_kendall_tau:
	metrics = [TopPercentMatch(p=5), KendallTauCorr()]
else:
	metrics = [TopPercentMatch(p=5)]

split_perf_groups = [('val', 'test_acc')]
if args.include_train_results:
	split_perf_groups += [('train', 'train_acc')]

embed_key = 'embed_beta'
logistic_key = 'logistic_beta'
runs = ['normal_B', 'normal_C', 'normal_D', 'normal_E', 'normal_F']

all_results = {}
for dataset in datasets:
	dat = load_dataset_dict(dataset, embed_key, logistic_key)
	for metric in metrics:
		for split, perf in split_perf_groups:
			stats = fetch_stats(dat, split, metric, 'normal_A', runs, 'uniform', embed_key, logistic_key, perf)
			all_results[(dataset, metric.name, split, perf)] = stats
			print('%s, %s, %s, %s' % (dataset, metric.name, split, perf))
			print(stats)

with open(out_file, 'wb+') as f:
	pkl.dump(all_results, f)



