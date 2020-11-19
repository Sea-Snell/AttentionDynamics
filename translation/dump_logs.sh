#!/bin/bash

python eval_dump.py --dataset multi30k --eval_bleu
#python eval_dump.py --dataset iwslt14 --eval_bleu
python eval_dump.py --dataset news_commentary_v14_en_nl --eval_bleu
#python eval_dump.py --dataset news_commentary_v14_en_pt --eval_bleu
#python eval_dump.py --dataset news_commentary_v14_it_pt --eval_bleu
