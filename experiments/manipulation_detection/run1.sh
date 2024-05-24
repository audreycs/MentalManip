#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model roberta-base --epoch 50 --train_batch_size 48 --train_data Dreaddit
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model roberta-base --epoch 50 --train_batch_size 32 --train_data SDCNL
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model roberta-base --epoch 50 --train_batch_size 64 --train_data ToxiGen
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model roberta-base --epoch 50 --train_batch_size 32 --train_data DetexD
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model roberta-base --epoch 50 --train_batch_size 32 --train_data fox-news
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model roberta-base --epoch 50 --train_batch_size 32 --train_data toxichat
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model roberta-base --epoch 50 --train_batch_size 32 --train_data MDRDC
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model roberta-base --epoch 50 --train_batch_size 32 --train_data mentalmanip