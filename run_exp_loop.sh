#!/bin/bash

# Define the values you want to loop through
PROMPT_LENGTHS=(8192 16384 32768) # 8K, 16K, and 32K - 128
PARALLEL_SAMPLES=(1 2 4 8 16 32 64 128 256 512 1024 2048)
BIFURCATED_ATTN=(0 1)

# Adjust the upper limit for prompt length
PROMPT_LENGTHS[2]=$((32 * 1024 - 128))

# export GROUP_NAME="compare_bifurcated_v3"
export GROUP_NAME="compare_bifurcated_v4"
export CUDA_VISIBLE_DEVICES=7

# Loop through the configurations
for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        for bifurcated_attn in "${BIFURCATED_ATTN[@]}"; do
            # Without compile
            python generate.py --wandb_group $GROUP_NAME --prompt_len $prompt_len --parallel_samples $parallel_sample --bifurcated_attn $bifurcated_attn --gqa_aware 1

            # With compile
            python generate.py --wandb_group $GROUP_NAME --prompt_len $prompt_len --parallel_samples $parallel_sample --bifurcated_attn $bifurcated_attn --compile --gqa_aware 1
        done
    done
done
