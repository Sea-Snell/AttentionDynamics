#!/bin/bash

python eval_dump.py --dataset multi30k --eval_bleu --include_train_subset --grad_bsize 32
# python eval_dump.py --dataset iwst14 --eval_bleu --include_train_subset --grad_bsize 32
python eval_dump.py --dataset news_commentary_v14_en_nl --eval_bleu --include_train_subset --grad_bsize 32
# python eval_dump.py --dataset news_commentary_v14_en_pt --eval_bleu --include_train_subset --grad_bsize 32
python eval_dump.py --dataset news_commentary_v14_it_pt --eval_bleu --include_train_subset --grad_bsize 32
