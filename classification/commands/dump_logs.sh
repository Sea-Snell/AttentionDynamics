#!/bin/bash

python ../eval_dump.py --dataset IMDB
python ../eval_dump.py --dataset Furd
python ../eval_dump.py --dataset Newsgroups
python ../eval_dump.py --dataset AG_News
python ../eval_dump.py --dataset SMS
python ../eval_dump.py --dataset Amzn
python ../eval_dump.py --dataset Yelp
