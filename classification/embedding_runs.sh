#!/bin/bash 

python embedding_beta.py --dataset IMDB --epochs 60 --lr 0.00005
python embedding_beta.py --dataset Furd --epochs 60 --lr 0.00015
python embedding_beta.py --dataset AG_News --epochs 60 --lr 0.00015
python embedding_beta.py --dataset Newsgroups --epochs 60 --lr 0.00015
python embedding_beta.py --dataset Yelp --epochs 60 --lr 0.00005
python embedding_beta.py --dataset Amzn --epochs 60 --lr 0.00005
python embedding_beta.py --dataset SMS --epochs 60 --lr 0.00015
