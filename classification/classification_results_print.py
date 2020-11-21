import pickle as pkl

with open('classification_results.pkl', 'rb') as f:
	dat = pkl.load(f)

def numbers_str(dict_, round_=True):
	def to_str(item, multiplier=1.0):
		if item is None:
			return 'null'
		if round_:
			return '%0.0f' % (item * multiplier)
		return '%0.2f' % (item * multiplier)
	items = (to_str(dict_['agr_unif'], 100), to_str(dict_['agr_px'], 100), to_str(dict_['agr_grad'], 100), to_str(dict_['baseline'], 100), to_str(dict_['xi_unif'], 100), to_str(dict_['xi_px'], 100), to_str(dict_['xi_grad'], 100), to_str(dict_['best_perf'], 100))
	items_str = "%s & %s & %s & %s & %s & %s & %s & %s" % items
	return items_str

items = [('IMDB', 'top 5% match', 'train', 'train_acc'),
		('Yelp', 'top 5% match', 'train', 'train_acc'),
		('SMS', 'top 5% match', 'train', 'train_acc'),
		('AG_News', 'top 5% match', 'train', 'train_acc'),
		('Newsgroups', 'top 5% match', 'train', 'train_acc'),
		('Furd', 'top 5% match', 'train', 'train_acc'),
		('Amzn', 'top 5% match', 'train', 'train_acc')]

for item in items:
	print(item[0] + '(C) & ' + numbers_str(dat[item], round_=False) + ' \\\\')