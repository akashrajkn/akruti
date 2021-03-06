#!/bin/bash

MODELID=11

for VARIABLE in arabic finnish georgian german hungarian maltese navajo russian spanish turkish
do
    srun --gres=gpu:1 -t 15:00:00 python main.py  --train -language=$VARIABLE -model_id=$MODELID -epochs=200 -batch_size=32
    srun --gres=gpu:1 -t 15:00:00 python main.py  --test  -language=$VARIABLE -model_id=$MODELID
                                  python evalm.py         -language=$VARIABLE -model_id=$MODELID
done
