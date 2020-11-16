import numpy as np
import math
from scipy import stats

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
            random_beta = np.array(beta)
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
