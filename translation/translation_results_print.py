import pickle as pkl

with open('multi30k_results.pkl', 'rb') as f:
	dat = pkl.load(f)

def numbers_str(dict_, round_=True):
	items = (100*dict_['agr_unif'], 100*dict_['agr_px'], 100*dict_['agr_grad'], 100*dict_['baseline'], dict_['xi_unif'], dict_['xi_px'], dict_['xi_grad'], dict_['best_perf'])
	if round_:
		items_str = "%0.0f & %0.0f & %0.0f & %0.0f & %0.0f & %0.0f & %0.0f & %0.0f" % items
	else:
		items_str = "%0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f" % items
	return items_str

print(numbers_str(dat[('multi30k', 'kendall tau', 'val', 'val_bleu')], round_=False))