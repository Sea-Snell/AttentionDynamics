#!/bin/bash

python train_classifier.py --dataset IMDB --seed 20 --steps 4000 --save_every 250  --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset IMDB --seed 13 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset IMDB --seed 12 --uniform --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset IMDB --seed 14 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset IMDB --seed 15 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset IMDB --seed 16 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset IMDB --seed 17 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250

python train_classifier.py --dataset Furd --seed 20 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Furd --seed 13 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Furd --seed 12 --uniform --steps 4000 --save_every 250 --custom_saves 10,50,100,150,200,250
python train_classifier.py --dataset Furd --seed 14 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Furd --seed 15 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Furd --seed 16 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Furd --seed 17 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250

python train_classifier.py --dataset Newsgroups --seed 20 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Newsgroups --seed 13 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Newsgroups --seed 12 --uniform --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Newsgroups --seed 14 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Newsgroups --seed 15 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Newsgroups --seed 16 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Newsgroups --seed 17 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250

python train_classifier.py --dataset AG_News --seed 20 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset AG_News --seed 13 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset AG_News --seed 12 --uniform --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset AG_News --seed 14 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset AG_News --seed 15 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset AG_News --seed 16 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset AG_News --seed 17 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250

python train_classifier.py --dataset SMS --seed 20 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset SMS --seed 13 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset SMS --seed 12 --uniform --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset SMS --seed 14 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset SMS --seed 15 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset SMS --seed 16 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset SMS --seed 17 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250

python train_classifier.py --dataset Amzn --seed 20 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Amzn --seed 13 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Amzn --seed 12 --uniform --steps 4000 --save_every 250 --custom_saves 10,50,100,150,200,250
python train_classifier.py --dataset Amzn --seed 14 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Amzn --seed 15 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Amzn --seed 16 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Amzn --seed 17 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250


python train_classifier.py --dataset Yelp --seed 20 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Yelp --seed 13 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Yelp --seed 12 --uniform --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Yelp --seed 14 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Yelp --seed 15 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Yelp --seed 16 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250
python train_classifier.py --dataset Yelp --seed 17 --steps 4000 --save_every 250 --custom_saves 0,10,50,100,150,200,250


