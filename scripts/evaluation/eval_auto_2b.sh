#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

dataset=$1

model_gnd_path="model_zoo/VideoMind-2B"
model_ver_path="model_zoo/VideoMind-2B"
model_pla_path="model_zoo/VideoMind-2B"

pred_path="outputs/$dataset"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_auto.py \
        --dataset $dataset \
        --pred_path $pred_path \
        --model_gnd_path $model_gnd_path \
        --model_ver_path $model_ver_path \
        --model_pla_path $model_pla_path \
        --chunk $CHUNKS \
        --index $IDX &
done

wait

python videomind/eval/eval_auto.py $pred_path --dataset $dataset
