# Understanding How Attention Learns in LSTM-based Machine Translation Models
code for experiments. <br />

this repo is split into classification and translation experiments in seperate folders. The file structure for both is roughly the same.


## classification
The main files to run to reproduce our classification experiments are as follows: <br />

Run "classification/train_classifier.py" to train a translation model <br />
This will save model checkpoints in the "models" directory <br />
Arguments: <br />
	--seed - provide a seed for training [required] <br />
	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"] <br />
	--steps - number of steps to train for [default = 2000] <br />
	--uniform - if this flag is used the model will train with uniform attention <br />
	--config - path to model config file [default = 'configs/model.json'] <br />
	--test_set_size - maximum number of datapoints to run validation on [detault = 4000] <br />
	--save_every - how often to checkpoint the model [default = 250] <br />
	--custom_saves - comma seperated list (no spaces!) of additional iterations to checkpoint [default = None] <br />

Run "classification/eval_dump.py" to collect alpha and beta statistics from saved model checkpoints <br />
This will output a pickle file with this data to the "outputs" directory <br />
Arguments: <br />
	--config - path to model config file [default = 'configs/model.json'] <br />
	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"] <br />
	--test_set_size - maximum number of datapoints to run validation on [detault = 4000] <br />
<br />
This script looks in the configs folder for the file named after the dataset is is called with. These configs sepecify a list which checkpoints to load based on iteration number, uniform attention or not, seed, and they provide a reference name for the model each checkpoint comes from. I'd recommend sticking to the naming convention of "Normal_A", "Normal_B", ..., "uniform" currently used because the current collect_classification_stats.py file depends on this naming convention. <br />

Run "classification/embedding_beta.py" to train a baseline embedding model <br />
This will output the learned beta to a pickle file in the "outputs" directory <br />
Arguments: <br />
	--dataset - name of which dataset to train on [required, one of "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"] <br />
	--bsize - batch size to train with [default = 64] <br />
	--epochs - number of epochs to train on [default = 30] <br />
	--embed_dim - dimention of embeddings [default = 256] <br />
	--test_set_size - maximum number of datapoints to run validation on [default = 4000] <br />
	--lr - learning rate for Adam Optimizer [default = 0.0002] <br />

Lastly run "classification/collect_classification_stats.py" to synthesize the correlation metrics that we report in our paper <br />
This will produce a pickle file containing a dictionary of all the results <br />
Arguments: <br />
	--datasets - space seperated list of dataset names [required, must be from: "IMDB", "Furd", "AG_News", "Newsgroups", "Yelp", "Amzn", "SNS"] <br />
	--out_file - name and path of file to output stats to [required] <br />
	--include_kendall_tau - call with this flag to also get kendall tau correlations <br />
	--include_train_results - call with this flag to also get correlations on training data <br />

The output pickle file will be structured as a dictionary with keys identifying the different experiments. <br />
The keys are a tuple of (dataset name, metric name ["kendall tau" or "top 5% match"], split ["train" or "val"], performance metric used ["test_acc" or "train_acc"]) <br />
Within each of these keys will be a dictionary with all the different results reported in our tables <br />
"agr_unif" is A(\alpha, \beta^{uf}) <br />
"agr_px" is A(\beta^{uf}, \beta^{px}) <br />
"arg_grad" is A(\Delta, \beta^{uf}) <br />
"agr_normal" is A(\alpha, \beta) <br />
"baseline" is \hat{A} <br />
"xi_unif" is \xi(\alpha, \beta^{uf}) <br />
"xi_px" is \xi(\beta^{uf}, \beta^{px}) <br />
"xi_grad" is \xi(\Delta, \beta^{uf}) <br />
"xi_normal" is \xi(\alpha, \beta) <br />
"best_perf" \xi^* <br />

<br />
We have provided two bash script files "classification/commands/dump_logs.sh" and "classification/commands/train_runs.sh", these run the exact set of "classification/train_classifier.py" and "classification/eval_dump.py" commands we ran. You will have to run "classification/embedding_beta.py" and "classification/collect_classification_stats.py" seperately. <br />
<br />

## translation
The main files for to run to reproduce our translation experiments are as follows: <br />

Run "translation/train_translation.py" to train a translation model <br />
This will save model checkpoints in the "models" directory <br />
Arguments: <br />
	--seed - provide a seed for training [required] <br />
	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"] <br />
	--epochs - number of epochs to train for [default = 20] <br />
	--uniform - if this flag is used the model will train with uniform attention <br />
	--config - path to model config file [default = 'configs/model.json'] <br />
	--save_every - how often to checkpoint the model [default = 250] <br />
	--custom_saves - comma seperated list (no spaces!) of additional iterations to checkpoint [default = None] <br />

Run "translation/eval_dump.py" to collect alpha and beta statistics from saved model checkpoints <br />
This will output a pickle file with this data to the "outputs" directory <br />
Arguments: <br />
	--config - path to model config file [default = 'configs/model.json'] <br />
	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"] <br />
	--eval_bleu - if this flag is used, it will evalute bleu as well as token accuracy <br />
	--include_train_subset - if this flag is called it will evaluate on a subset of training data equivalent to the size of the validation set <br />
	--grad_bsize - this sets the batchsize for computing the gradient influence score [default = 16] <br />
<br />
This script looks in the configs folder for the file named after the dataset is is called with. These configs sepecify a list which checkpoints to load based on iteration number, uniform attention or not, seed, whether to evaluate gradient influence on this checkpoint, and they provide a reference name for the model each checkpoint comes from. I'd recommend sticking to the naming convention of "Normal_A", "Normal_B", ..., "uniform" currently used because the current collect_translation_stats.py file depends on this naming convention. <br />

Run "translation/embedding_beta.py" to train a baseline embedding model <br />
This will output the learned beta to a pickle file in the "outputs" directory <br />
Arguments: <br />
	--dataset - name of which dataset to train on [required, one of "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"] <br />
	--bsize - batch size to train with [default = 64] <br />
	--epochs - number of epochs to train on [default = 30] <br />
	--embed_dim - dimention of embeddings [default = 256] <br />

Lastly run "translation/collect_classification_stats.py" to synthesize the correlation metrics that we report in our paper
This will produce a pickle file containing a dictionary of all the results <br />
Arguments: <br />
	--datasets - space seperated list of dataset names [required, must be from: "multi30k", "iwslt14", "news_commentary_v14_en_nl", "news_commentary_v14_en_pt", "news_commentary_v14_it_pt"] <br />
	--out_file - name and path of file to output stats to [required] <br />
	--include_kendall_tau - call with this flag to also get kendall tau correlations <br />
	--include_train_results - call with this flag to also get correlations on training data <br />
	--include_bleu_results - call with flag to also get correlations using bleu performance metric <br />

The output pickle file will be structured as a dictionary with keys identifying the different experiments. <br />
The keys are a tuple of (dataset name, metric name ["kendall tau" or "top 5% match"], split ["train" or "val"], performance metric used ["val_acc", "train_acc", "val_bleu", or "train_bleu"]) <br />
Within each of these keys will be a dictionary with all the different results reported in our tables <br />
"agr_unif" is A(\alpha, \beta^{uf}) <br />
"agr_px" is A(\beta^{uf}, \beta^{px}) <br />
"arg_grad" is A(\Delta, \beta^{uf}) <br />
"agr_normal" is A(\alpha, \beta) <br />
"baseline" is \hat{A} <br />
"xi_unif" is \xi(\alpha, \beta^{uf}) <br />
"xi_px" is \xi(\beta^{uf}, \beta^{px}) <br />
"xi_grad" is \xi(\Delta, \beta^{uf}) <br />
"xi_normal" is \xi(\alpha, \beta) <br />
"best_perf" \xi^* <br />

<br />

We have provided two bash script files "translation/commands/dump_logs.sh" and "translation/commands/train_runs.sh", these run the exact set of "translation/train_classifier.py" and "translation/eval_dump.py" commands we ran. You will have to run "translation/embedding_beta.py" and "translation/collect_classification_stats.py" seperately.

## Sequence Copying

python3 seq2seq.py --no_rep

To compare distribution of permutation of 40 vocabs vs. 40 out of 60, run
"python3 seq2seq.py --no_rep", 
and 
"python3 seq2seq.py --no_rep num_vocab=60"

To reproduce the multi-head attention experiment in Section 5.2, run 
"python3 multihead.py". The learning curves will be dumped into re_8headcomb.pkl
