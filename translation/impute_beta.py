import pickle as pkl
import numpy as np

embed_dim = 256
dataset_name = 'iwslt14'#'multi30k'
dict_f_path = 'data/{dataset_name}_logs_rz.pkl'.format(dataset_name=dataset_name)
dicts = pkl.load(open(dict_f_path, 'rb'))
embed_beta = pkl.load(open('data/{dataset_name}embedding{embed_dim}translation.pkl'.format(embed_dim=embed_dim, dataset_name=dataset_name), 'rb'))
for d in dicts:
    src, trg = d['src'], d['trg']
    betas = []
    for t in trg[1:]:
        beta = []
        for s in src:
            beta.append(embed_beta[s][t])
        betas.append(beta)
    betas = np.array(betas)
    d['embed{embed_dim}_beta'.format(embed_dim=embed_dim)] = betas
# print(dicts[0]['embed{embed_dim}_beta'.format(embed_dim=embed_dim)])
pkl.dump(dicts, open(dict_f_path, 'wb'))
