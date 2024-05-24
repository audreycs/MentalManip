#!/usr/bin/env bash

# Example command lines to run the experiments

# for zero-shot prompting using chatgpt and llama-13b on mentalmanip_maj dataset
python zeroshot_prompt.py --model chatgpt --data ../datasets/mentalmanip_con.csv
CUDA_VISIBLE_DEVICES=0,1 python zeroshot_prompt.py --model llama-13b --data ../datasets/mentalmanip_con.csv

# for few-shot prompting using chatgpt and llama-13b on mentalmanip_maj dataset
python fewshot_prompt.py --model chatgpt --data ../datasets/mentalmanip_con.csv
CUDA_VISIBLE_DEVICES=0,1 python fewshot_prompt.py --model llama-13b --data ../datasets/mentalmanip_con.csv

# for fine-tuning llama-13b on mentalmanip and other datasets (first train and then evaluate)
CUDA_VISIBLE_DEVICES=0,1 python finetune.py --model llama-13b --mode train --eval_data mentalmanip_con --train_data Dreaddit
CUDA_VISIBLE_DEVICES=0,1 python finetune.py --model llama-13b --mode eval --eval_data mentalmanip_con --train_data Dreaddit

CUDA_VISIBLE_DEVICES=0,1 python finetune.py --model llama-13b --mode train --eval_data mentalmanip_con --train_data mentalmanip
CUDA_VISIBLE_DEVICES=0,1 python finetune.py --model llama-13b --mode eval --eval_data mentalmanip_con --train_data mentalmanip

# for fine-tuning roberta-base on mentalmanip and other datasets
CUDA_VISIBLE_DEVICES=0,1 python finetune.py --model roberta-base --epoch 50 --train_batch_size 16 --mode train --eval_data mentalmanip_con --train_data Dreaddit
CUDA_VISIBLE_DEVICES=0,1 python finetune.py --model roberta-base --epoch 50 --train_batch_size 16 --mode train --eval_data mentalmanip_con --train_data Dreaddit
