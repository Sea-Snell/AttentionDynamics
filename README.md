# Approximating How Single Head Attention Learns <br />

this repo is split into classification and translation experiments in seperate folders. The file structure for both is roughly the same.

This repo has been tested on python3.7

Run ```pip install -r requirements.txt``` to install dependencies

## classification

The main steps to reproduce our classification experiments are as follows:

For all of these steps make sure you are in the ```classification/``` directory, so first ```cd classification/```

Download our processed data files [here](https://drive.google.com/drive/folders/1erdWgIK8wxypwgOZScy8uB7yDWFRzVoU?usp=sharing). Make sure this folder in named ```data``` and place it in the ```classification/``` directory


Run ```python train_classifier.py``` to train a classification model. This will save model checkpoints in the ```models/``` directory.

Arguments:

	--seed - provide a seed for training [required]
	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"]
	--steps - number of steps to train for [default = 2000]
	--uniform - if this flag is used the model will train with uniform attention
	--config - path to model config file [default = 'configs/model.json']
	--test_set_size - maximum number of datapoints to run validation on [detault = 4000]
	--save_every - how often to checkpoint the model [default = 250]
	--custom_saves - comma seperated list (no spaces!) of additional iterations to checkpoint [default = None]

Run ```python eval_dump.py``` to collect alpha and beta statistics from saved model checkpoints. This will output a pickle file with this data to the ```outputs/``` directory.

Arguments:

	--config - path to model config file [default = 'configs/model.json']
	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"]
	--test_set_size - maximum number of datapoints to run validation on [detault = 4000]

This script looks in ```configs/``` for the file named after the dataset that it is called with. These configs specify a list of checkpoints to load based on iteration number, uniform attention or not, seed, and a reference name for the model each checkpoint comes from. I'd recommend sticking to the naming convention of "Normal_A", "Normal_B", ..., "uniform" currently used because the current collect_classification_stats.py file depends on this naming convention.

Run ```python embedding_beta.py``` to train a baseline embedding model. This will output the learned beta to a pickle file in the ```outputs/``` directory.

Arguments:

	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"]
	--bsize - batch size to train with [default = 64]
	--epochs - number of epochs to train on [default = 30]
	--embed_dim - dimention of embeddings [default = 256]
	--test_set_size - maximum number of datapoints to run validation on [default = 4000]
	--lr - learning rate for Adam Optimizer [default = 0.0002]

Lastly run ```python classification/collect_classification_stats.py``` to synthesize the correlation metrics that we report in our paper. This will produce a pickle file containing a dictionary of all the results.

Arguments:

	--datasets - space seperated list of dataset names [required, must be from: "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"]
	--out_file - name and path of file to output stats to [required]
	--include_kendall_tau - call with this flag to also get kendall tau correlations
	--include_train_results - call with this flag to also get correlations on training data

The output pickle file will be structured as a dictionary with keys identifying the different experiments. <br />
The keys are a tuple of (dataset name, metric name ["kendall tau" or "top 5% match"], split ["train" or "val"], performance metric used ["test_acc" or "train_acc"]) <br />
Within each of these keys will be a dictionary with all the different results reported in our tables.

[comment]: <> (\mathcal{A}&#40;\alpha, \beta^{uf}&#41;)
[comment]: <> (\mathcal{A}&#40;\beta^{uf}, \beta^{px}&#41;)
[comment]: <> (\mathcal{A}&#40;\Delta, \beta^{uf}&#41;)
[comment]: <> (\mathcal{A}&#40;\alpha, \beta&#41;)
[comment]: <> (\hat{\mathcal{A}})
[comment]: <> (\xi&#40;\alpha, \beta^{uf}&#41;)
[comment]: <> (\xi&#40;\beta^{uf}, \beta^{px}&#41;)
[comment]: <> (\xi&#40;\Delta, \beta^{uf}&#41;)
[comment]: <> (\xi&#40;\alpha, \beta&#41;)
[comment]: <> (\xi^*)

- "agr_unif" is ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%28%5Calpha%2C%20%5Cbeta%5E%7Buf%7D%29)
- "agr_px" is ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%28%5Cbeta%5E%7Buf%7D%2C%20%5Cbeta%5E%7Bpx%7D%29)
- "arg_grad" is ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%28%5CDelta%2C%20%5Cbeta%5E%7Buf%7D%29)
- "agr_normal" is ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%28%5Calpha%2C%20%5Cbeta%29)
- "baseline" is ![](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmathcal%7BA%7D%7D)
- "xi_unif" is ![](https://latex.codecogs.com/gif.latex?%5Cxi%28%5Calpha%2C%20%5Cbeta%5E%7Buf%7D%29)
- "xi_px" is ![](https://latex.codecogs.com/gif.latex?%5Cxi%28%5Calpha%2C%20%5Cbeta%5E%7Buf%7D%29)
- "xi_grad" is ![](https://latex.codecogs.com/gif.latex?%5Cxi%28%5CDelta%2C%20%5Cbeta%5E%7Buf%7D%29)
- "xi_normal" is ![](https://latex.codecogs.com/gif.latex?%5Cxi%28%5Calpha%2C%20%5Cbeta%29)
- "best_perf" ![](https://latex.codecogs.com/gif.latex?%5Cxi%5E*)


We have provided two bash script files ```dump_logs.sh``` and ```train_runs.sh```, these run the exact set of ```train_classifier.py``` and ```eval_dump.py``` commands we ran. You will have to run ```embedding_beta.py``` and ```collect_classification_stats.py``` separately.

## translation

The main steps to reproduce our translation experiments are as follows:

For all of these steps make sure you are in the ```translation/``` directory, so first ```cd translation/```

Download our processed data files [here](https://drive.google.com/drive/folders/1b_Oyxci1EpOK-LJc4u0IJ-lmtWsu9dK3?usp=sharing). Make sure this folder in named ```data``` and place it in the ```translation/``` directory

Run ```python train_translation.py``` to train a translation model. This will save model checkpoints in the ```models/``` directory.

Arguments:

	--seed - provide a seed for training [required]
	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"]
	--epochs - number of epochs to train for [default = 20]
	--uniform - if this flag is used the model will train with uniform attention
	--config - path to model config file [default = 'configs/model.json']
	--save_every - how often to checkpoint the model [default = 250]
	--custom_saves - comma seperated list (no spaces!) of additional iterations to checkpoint [default = None]

Run ```python eval_dump.py``` to collect alpha and beta statistics from saved model checkpoints. This will output a pickle file with this data to the ```outputs/``` directory.

Arguments:

	--config - path to model config file [default = 'configs/model.json']
	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"]
	--eval_bleu - if this flag is used, it will evalute bleu as well as token accuracy
	--include_train_subset - if this flag is called it will evaluate on a subset of training data equivalent to the size of the validation set
	--grad_bsize - this sets the batchsize for computing the gradient influence score [default = 16]


This script looks in the configs folder for the file named after the dataset that it is called with. These configs specify a list which checkpoints to load based on iteration number, uniform attention or not, seed, whether to evaluate gradient influence on this checkpoint, and they provide a reference name for the model each checkpoint comes from. I'd recommend sticking to the naming convention of "Normal_A", "Normal_B", ..., "uniform" currently used because the current collect_translation_stats.py file depends on this naming convention.

Run ```python embedding_beta.py``` to train a baseline embedding model. This will output the learned beta to a pickle file in the ```outputs/``` directory

Arguments:

	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"]
	--bsize - batch size to train with [default = 64]
	--epochs - number of epochs to train on [default = 30]
	--embed_dim - dimention of embeddings [default = 256]

Lastly run ```python collect_classification_stats.py``` to synthesize the correlation metrics that we report in our paper. This will produce a pickle file containing a dictionary of all the results.

Arguments:

	--datasets - space seperated list of dataset names [required, must be from: "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"]
	--out_file - name and path of file to output stats to [required]
	--include_kendall_tau - call with this flag to also get kendall tau correlations
	--include_train_results - call with this flag to also get correlations on training data
	--include_bleu_results - call with flag to also get correlations using bleu performance metric

The output pickle file will be structured as a dictionary with keys identifying the different experiments.
The keys are a tuple of (dataset name, metric name ["kendall tau" or "top 5% match"], split ["train" or "val"], performance metric used ["val_acc", "train_acc", "val_bleu", or "train_bleu"]).
Within each of these keys will be a dictionary with all the different results reported in our tables

[comment]: <> (\mathcal{A}&#40;\alpha, \beta^{uf}&#41;)
[comment]: <> (\mathcal{A}&#40;\beta^{uf}, \beta^{px}&#41;)
[comment]: <> (\mathcal{A}&#40;\Delta, \beta^{uf}&#41;)
[comment]: <> (\mathcal{A}&#40;\alpha, \beta&#41;)
[comment]: <> (\hat{\mathcal{A}})
[comment]: <> (\xi&#40;\alpha, \beta^{uf}&#41;)
[comment]: <> (\xi&#40;\beta^{uf}, \beta^{px}&#41;)
[comment]: <> (\xi&#40;\Delta, \beta^{uf}&#41;)
[comment]: <> (\xi&#40;\alpha, \beta&#41;)
[comment]: <> (\xi^*)

- "agr_unif" is ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%28%5Calpha%2C%20%5Cbeta%5E%7Buf%7D%29)
- "agr_px" is ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%28%5Cbeta%5E%7Buf%7D%2C%20%5Cbeta%5E%7Bpx%7D%29)
- "arg_grad" is ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%28%5CDelta%2C%20%5Cbeta%5E%7Buf%7D%29)
- "agr_normal" is ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BA%7D%28%5Calpha%2C%20%5Cbeta%29)
- "baseline" is ![](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmathcal%7BA%7D%7D)
- "xi_unif" is ![](https://latex.codecogs.com/gif.latex?%5Cxi%28%5Calpha%2C%20%5Cbeta%5E%7Buf%7D%29)
- "xi_px" is ![](https://latex.codecogs.com/gif.latex?%5Cxi%28%5Calpha%2C%20%5Cbeta%5E%7Buf%7D%29)
- "xi_grad" is ![](https://latex.codecogs.com/gif.latex?%5Cxi%28%5CDelta%2C%20%5Cbeta%5E%7Buf%7D%29)
- "xi_normal" is ![](https://latex.codecogs.com/gif.latex?%5Cxi%28%5Calpha%2C%20%5Cbeta%29)
- "best_perf" ![](https://latex.codecogs.com/gif.latex?%5Cxi%5E*)


We have provided two bash script files ```dump_logs.sh``` and ```train_runs.sh```, these run the exact set of ```train_classifier.py``` and ```eval_dump.py``` commands we ran. You will have to run ```embedding_beta.py``` and ```collect_classification_stats.py``` separately.

## Sequence Copying

Make sure you are in the ```SequenceCopying/``` directory, so first ```cd SequenceCopying/```

```python seq2seq.py --no_rep```

Arguments:

	--no_rep - call with this flag to prevent sequences with repeated tokens
	--num_vocab - specifies the vocab size [default = 40]

To compare distribution of permutation of 40 vocabs vs. 40 out of 60, run
```python seq2seq.py --no_rep```, 
and 
```python seq2seq.py --no_rep num_vocab=60```

To reproduce the multi-head attention experiment in Section 5.2, run 
```python multihead.py```. The learning curves will be dumped into re_8headcomb.pkl
