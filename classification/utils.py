import numpy as np
import math
from scipy import stats
import pickle as pkl

num_random = 1

class Metric:

    def __init__(self, name):
        self.name = name

    def eval_corr_single(self, alpha, beta):
        raise NotImplementedError

    def eval_corr(self, dicts, key1, key2):
        total, corr_sum, baseline_sum = 0, 0., 0.
        # for each datapoints
        for d in dicts:
            alpha, beta = d[key1], d[key2]
            assert len(alpha) == len(beta)
            # for each target word
            corr = self.eval_corr_single(alpha, beta)
            if np.isnan(corr):
                continue
            total += 1
            corr_sum += corr

            # obtain the random baseline by shuffling beta
            random_beta = np.copy(np.array(beta))
            total_random_corr_single = 0.
            for _ in range(num_random):
                np.random.shuffle(random_beta)
                total_random_corr_single += self.eval_corr_single(alpha, random_beta)
            baseline_sum += total_random_corr_single / num_random

        return {'name': self.name, 'correlation': corr_sum / total, 'baseline': baseline_sum / total}


def topKMatch_single(alpha, beta, k):
    assert k > 0
    if np.sum(np.array(beta) > beta[np.argmax(alpha)]) < k:
        return 1
    else:
        return 0


class TopKMatch(Metric):

    def __init__(self, k):
        name = 'top {k} match'.format(k=k)
        super().__init__(name)
        self.k = k

    def eval_corr_single(self, alpha, beta):
        return topKMatch_single(alpha, beta, self.k)


class TopPercentMatch(Metric):

    def __init__(self, p):
        name = 'top {p}% match'.format(p=p)
        super().__init__(name)
        self.p = float(p) / 100

    def eval_corr_single(self, alpha, beta):
        return topKMatch_single(alpha, beta, math.ceil(self.p * len(beta)))


class SpearmanRankCorr(Metric):

    def __init__(self):
        name = 'spearman rank'
        super().__init__(name)

    def eval_corr_single(self, alpha, beta):
        corr, _ = stats.spearmanr(alpha, beta)
        return corr


class KendallTauCorr(Metric):
    def __init__(self):
        name = 'kendall tau'
        super().__init__(name)

    def eval_corr_single(self, alpha, beta):
        corr, _ = stats.kendalltau(alpha, beta)
        return corr




def passing_idx(A1s, A2):
    for i in range(len(A1s)):
        if A1s[i] > A2:
            return i
    return None

def corrs_iter(dicts, key1, keys2, corr_metric, reverse=False):
    corrs = []
    baselines = []
    for key2 in keys2:
        if reverse:
                vals = corr_metric.eval_corr(dicts, key2, key1)
        else:
            vals = corr_metric.eval_corr(dicts, key1, key2)
        corrs.append(vals['correlation'])
        baselines.append(vals['baseline'])
    return corrs, baselines

def acc_iter(metas, keys):
    accs = []
    for key in keys:
        accs.append(metas[key])
    return accs

def flip_betas(dicts):
    for item in dicts:
        for key in item:
            if type(key) is tuple and key[2] == 'beta' and item[key] is not None:
                item[key] = item[key] * (2 * item['trg'] - 1)

def flip_grads(dicts):
    for item in dicts:
        for key in item:
            if type(key) is tuple and key[2] == 'grad' and item[key] is not None:
                item[key] = -item[key]

def max_corr(dicts, key1, keys2, metric, reverse=False):
    return max(corrs_iter(dicts, key1, keys2, metric, reverse=reverse)[0])

def impute_beta(dicts, beta_vector, key_name):
    for item in dicts:
        beta = []
        for tok in item['src']:
            beta.append(beta_vector[tok] * (2 * item['trg'] - 1))
        item[key_name] = beta

def is_valid_dict(dict_):
    for k, v in dict_.items():
        if v is None:
            return False
    return True

def fetch_stats(dat, split, metric, gold_run_name, comparison_run_names, unif_run_name, px_name, logistic_name, performance_name):
    all_dicts = dat['data']
    dicts = [d for d in all_dicts if d['split'] == split and is_valid_dict(d)]
    iterations = sorted(list(set([key[1] for key in dat['metas']])))
    normalA_accs = acc_iter(dat['metas'], [(gold_run_name, iter_, performance_name) for iter_ in iterations])
    normalA_iter = iterations[np.argmax(normalA_accs)]
    gold_alpha_key = (gold_run_name, normalA_iter, 'alpha')
    gold_grad_key = (gold_run_name, normalA_iter, 'grad')
    
    avg_corr = None
    avg_perf = None
    baseline = None
    for run in comparison_run_names:
        normal_keys = [(run, iter_, 'alpha') for iter_ in iterations]
        acc_keys = [(run, iter_, performance_name) for iter_ in iterations]
        
        alpha_corrs, alpha_baseline = corrs_iter(dicts, gold_alpha_key, normal_keys, metric)
        alpha_perfs = acc_iter(dat['metas'], acc_keys)
        
        if avg_corr is None:
            avg_corr = np.array(alpha_corrs)
            avg_perf = np.array(alpha_perfs)
            baseline = sum(alpha_baseline) / len(alpha_baseline)
        else:
            avg_corr += np.array(alpha_corrs)
            avg_perf += np.array(alpha_perfs)
            baseline += (sum(alpha_baseline) / len(alpha_baseline))
    avg_corr = avg_corr / len(comparison_run_names)
    avg_perf = avg_perf / len(comparison_run_names)
    baseline = baseline / len(comparison_run_names)
    
    beta_unif_keys = [(unif_run_name, iter_, 'beta') for iter_ in iterations]
    beta_normal_keys = [(comparison_run_names[0], iter_, 'beta') for iter_ in iterations]
    beta_corr_unif = max_corr(dicts, gold_alpha_key, beta_unif_keys, metric)
    beta_corr_grad = max_corr(dicts, gold_grad_key, beta_unif_keys, metric)
    beta_corr_px = max_corr(dicts, px_name, beta_unif_keys, metric)
    beta_corr_logistic = max_corr(dicts, logistic_name, beta_unif_keys, metric)
    beta_corr_normal = max_corr(dicts, gold_alpha_key, beta_normal_keys, metric)
    
    best_acc = dat['metas'][(gold_run_name, normalA_iter, performance_name)]
    idx_unif = passing_idx(avg_corr, beta_corr_unif)
    idx_grad = passing_idx(avg_corr, beta_corr_grad)
    idx_px = passing_idx(avg_corr, beta_corr_px)
    idx_logistic = passing_idx(avg_corr, beta_corr_logistic)
    idx_normal = passing_idx(avg_corr, beta_corr_normal)
    
    def out_perf(idx):
        if idx is None:
            return None
        else:
            return avg_perf[idx]
    
    return {'agr_unif': beta_corr_unif, 'agr_px': beta_corr_px, 'agr_logistic': beta_corr_logistic, 'agr_grad': beta_corr_grad, 'agr_normal': beta_corr_normal, 'xi_unif': out_perf(idx_unif), 
            'xi_px': out_perf(idx_px), 'xi_logistic': out_perf(idx_logistic), 'xi_grad': out_perf(idx_grad), 'xi_normal': out_perf(idx_normal), 'best_perf': best_acc, 'baseline': baseline,
            'alpha_corrs': avg_corr, 'alpha_perfs': avg_perf, 'iterations': iterations}

def load_dataset_dict(dataset_name, embed_key, logistic_key):
    dat = pkl.load(open('outputs/{dataset}_logs.pkl'.format(dataset=dataset_name), 'rb'))
    embed_beta = pkl.load(open('outputs/{dataset}embedding256beta.pkl'.format(dataset=dataset_name), 'rb'))
    logistic_beta = pkl.load(open('outputs/{dataset}_logistic_beta.pkl'.format(dataset=dataset_name), 'rb'))
    impute_beta(dat['data'], embed_beta, embed_key)
    impute_beta(dat['data'], logistic_beta, logistic_key)
    flip_betas(dat['data'])
    flip_grads(dat['data'])
    return dat


