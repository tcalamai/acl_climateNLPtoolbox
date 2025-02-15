#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export HUB_TOKEN=""
python experiment.py --log performances_longformer_logicclimate --seed_list 42 -l --batch_size 5 --accumulation_steps 3 --dataset_list logicClimate