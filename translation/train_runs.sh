#!/bin/bash

# python train_translation.py --epochs 20 --dataset multi30k --seed 20 --save_every 2000 --config configs/model.json
# python train_translation.py --epochs 20 --dataset multi30k --seed 13 --save_every 2000 --config configs/model.json
# python train_translation.py --epochs 20 --dataset multi30k --seed 12 --save_every 2000 --uniform --config configs/model.json

# python train_translation.py --epochs 30 --dataset iwslt14 --seed 20 --save_every 2000 --config configs/model.json
# python train_translation.py --epochs 30 --dataset iwslt14 --seed 13 --save_every 2000 --config configs/model.json
# python train_translation.py --epochs 30 --dataset iwslt14 --seed 12 --save_every 2000 --uniform --config configs/model.json

#python train_translation.py --epochs 20 --dataset news_commentary_v14_en_nl --seed 20 --save_every 2000 --config configs/model.json --custom_saves 50,100,500,1000,1500
#python train_translation.py --epochs 20 --dataset news_commentary_v14_en_nl --seed 13 --save_every 2000 --config configs/model.json --custom_saves 50,100,500,1000,1500
#python train_translation.py --epochs 20 --dataset news_commentary_v14_en_nl --seed 12 --save_every 2000 --uniform --config configs/model.json --custom_saves 50,100,500,1000,1500

python train_translation.py --epochs 20 --dataset news_commentary_v14_en_pt --seed 20 --save_every 2000 --config configs/model.json --custom_saves 50,100,500,1000,1500
python train_translation.py --epochs 20 --dataset news_commentary_v14_en_pt --seed 13 --save_every 2000 --config configs/model.json --custom_saves 50,100,500,1000,1500
python train_translation.py --epochs 20 --dataset news_commentary_v14_en_pt --seed 12 --save_every 2000 --uniform --config configs/model.json --custom_saves 50,100,500,1000,1500

python train_translation.py --epochs 20 --dataset news_commentary_v14_it_pt --seed 20 --save_every 2000 --config configs/model.json --custom_saves 50,100,500,1000,1500
python train_translation.py --epochs 20 --dataset news_commentary_v14_it_pt --seed 13 --save_every 2000 --config configs/model.json --custom_saves 50,100,500,1000,1500
python train_translation.py --epochs 20 --dataset news_commentary_v14_it_pt --seed 12 --save_every 2000 --uniform --config configs/model.json --custom_saves 50,100,500,1000,1500
