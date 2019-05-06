#!/bin/bash
for VARIABLE in navajo arabic finnish german maltese 
do
    python preprocess.py -language=$VARIABLE --rewrite

    srun --gres=gpu:1 -t 2:00:00 python main.py --train -language=$VARIABLE -model_id=3 -epochs=180
    srun --gres=gpu:1            python main.py --test  -language=$VARIABLE -model_id=3
done
