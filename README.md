# Interpreting Attention Training in LSTMs with Classical Alignment
code for experiments.

this repo is split into classification and translation experiments in seperate folders. The file structure for both is roughly the same.


## classification
The main files to run to reproduce our classification experiments are as follows:

Run "classification/train_classifier.py" to train a translation model
This will save model checkpoints in the "models" directory
Arguments:
	--seed - provide a seed for training [required]
	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"]
	--steps - number of steps to train for [default = 2000]
	--uniform - if this flag is used the model will train with uniform attention
	--config - path to model config file [default = 'configs/model.json']
	--test_set_size - maximum number of datapoints to run validation on [detault = 4000]
	--save_every - how often to checkpoint the model [default = 250]
	--custom_saves - comma seperated list (no spaces!) of additional iterations to checkpoint [default = None]

Run "classification/eval_dump.py" to collect alpha and beta statistics from saved model checkpoints
This will output a pickle file with this data to the "outputs" directory
Arguments:
	--config - path to model config file [default = 'configs/model.json']
	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"]
	--test_set_size - maximum number of datapoints to run validation on [detault = 4000]
This script looks in the configs folder for the file named after the dataset is is called with. These configs sepecify a list which checkpoints to load based on iteration number, uniform attention or not, seed, and they provide a reference name for the model each checkpoint comes from. I'd recommend sticking to the naming convention of "Normal_A", "Normal_B", ..., "uniform" currently used because the current collect_classification_stats.py file depends on this naming convention.

Run "classification/embedding_beta.py" to train a baseline embedding model
This will output the learned beta to a pickle file in the "outputs" directory
Arguments:
	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"]
	--bsize - batch size to train with [default = 64]
	--epochs - number of epochs to train on [default = 30]
	--embed_dim - dimention of embeddings [default = 256]
	--test_set_size - maximum number of datapoints to run validation on [default = 4000]
	--lr - learning rate for Adam Optimizer [default = 0.0002]

Lastly run "classification/collect_classification_stats.py" to synthesize the correlation metrics that we report in our paper
This will produce a pickle file containing a dictionary of all the results
Arguments:
	--datasets - space seperated list of dataset names [required, must be from: "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"]
	--out_file - name and path of file to output stats to [required]
	--include_kendall_tau - call with this flag to also get kendall tau correlations
	--include_train_results - call with this flag to also get correlations on training data

The output pickle file will be structured as a dictionary with keys identifying the different experiments.
The keys are a tuple of (dataset name, metric name ["kendall tau" or "top 5% match"], split ["train" or "val"], performance metric used ["test_acc" or "train_acc"])
Within each of these keys will be a dictionary with all the different results reported in our tables
"agr_unif" is first column of table 1 in our paper
"agr_px" is the second column
"arg_grad" is the third column
"baseline" is the fourth column
"xi_unif" is the fith column
"xi_px" is the sixth column
"xi_grad" is the seventh column
"best_perf" is the last column

We have provided two bash script files "classification/commands/dump_logs.sh" and "classification/commands/train_runs.sh", these run the exact set of "classification/train_classifier.py" and "classification/eval_dump.py" commands we ran. You will have to run "classification/embedding_beta.py" and "classification/collect_classification_stats.py" seperately.


## translation
The main files for to run to reproduce our translation experiments are as follows:

Run "translation/train_translation.py" to train a translation model
This will save model checkpoints in the "models" directory
Arguments:
	--seed - provide a seed for training [required]
	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"]
	--epochs - number of epochs to train for [default = 20]
	--uniform - if this flag is used the model will train with uniform attention
	--config - path to model config file [default = 'configs/model.json']
	--save_every - how often to checkpoint the model [default = 250]
	--custom_saves - comma seperated list (no spaces!) of additional iterations to checkpoint [default = None]

Run "translation/eval_dump.py" to collect alpha and beta statistics from saved model checkpoints
This will output a pickle file with this data to the "outputs" directory
Arguments:
	--config - path to model config file [default = 'configs/model.json']
	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"]
	--eval_bleu - if this flag is used, it will evalute bleu as well as token accuracy
	--include_train_subset - if this flag is called it will evaluate on a subset of training data equivalent to the size of the validation set
	--grad_bsize - this sets the batchsize for computing the gradient influence score [default = 16]
This script looks in the configs folder for the file named after the dataset is is called with. These configs sepecify a list which checkpoints to load based on iteration number, uniform attention or not, seed, whether to evaluate gradient influence on this checkpoint, and they provide a reference name for the model each checkpoint comes from. I'd recommend sticking to the naming convention of "Normal_A", "Normal_B", ..., "uniform" currently used because the current collect_translation_stats.py file depends on this naming convention.

Run "translation/embedding_beta.py" to train a baseline embedding model
This will output the learned beta to a pickle file in the "outputs" directory
Arguments:
	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"]
	--bsize - batch size to train with [default = 64]
	--epochs - number of epochs to train on [default = 30]
	--embed_dim - dimention of embeddings [default = 256]

Lastly run "translation/collect_classification_stats.py" to synthesize the correlation metrics that we report in our paper
This will produce a pickle file containing a dictionary of all the results
Arguments:
	--datasets - space seperated list of dataset names [required, must be from: "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"]
	--out_file - name and path of file to output stats to [required]
	--include_kendall_tau - call with this flag to also get kendall tau correlations
	--include_train_results - call with this flag to also get correlations on training data
	--include_bleu_results - call with flag to also get correlations using bleu performance metric

The output pickle file will be structured as a dictionary with keys identifying the different experiments.
The keys are a tuple of (dataset name, metric name ["kendall tau" or "top 5% match"], split ["train" or "val"], performance metric used ["val_acc", "train_acc", "val_bleu", or "train_bleu"])
Within each of these keys will be a dictionary with all the different results reported in our tables
"agr_unif" is first column of table 1 in our paper
"agr_px" is the second column
"arg_grad" is the third column
"baseline" is the fourth column
"xi_unif" is the fith column
"xi_px" is the sixth column
"xi_grad" is the seventh column
"best_perf" is the last column

We have provided two bash script files "translation/commands/dump_logs.sh" and "translation/commands/train_runs.sh", these run the exact set of "translation/train_classifier.py" and "translation/eval_dump.py" commands we ran. You will have to run "translation/embedding_beta.py" and "translation/collect_classification_stats.py" seperately.




