#!/bin/bash

# Define the values you want to loop through
PROMPT_LENGTHS=(8192 16384 32768) # 8K, 16K, and 32K - 128
PARALLEL_SAMPLES=(1 2 4 8 16 32 64 128 256 512 1024 2048)
BIFURCATED_ATTN=(0 1)

# Adjust the upper limit for prompt length
PROMPT_LENGTHS[2]=$((32 * 1024 - 128))

# export GROUP_NAME="compare_bifurcated_v3"
export GROUP_NAME="compare_bifurcated"


for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash2
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 1 --compile
    done
done

echo "------------- Done with bifurcated compile -----------------------------"

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash2
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 1 
    done
done

echo "------------- Done with bifurcated normal -----------------------------"

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash2
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 0 --use_flash2_decode 1 --enable_sdpa_flash 0
    done
done

echo "------------- Done with flash 2 native -----------------------------"

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash2
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 0 --use_flash2_decode 0 --enable_sdpa_flash 0 
    done
done

echo "------------- Done with SDPA Math -----------------------------"

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash2
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 0 --use_flash2_decode 0 --enable_sdpa_flash 0 --compile 
    done
done

echo "------------- Done with SDPA Math + Compile -----------------------------"



for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash2
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_kv 1 --use_flash2_decode 1 
    done
done

echo "------------- Done with native flash 2 non contiguous mode -----------------------------"

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # flash via SDPA
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 0 --use_flash2_decode 0 --enable_sdpa_flash 1 
    done
done


echo "------------- Done with SDPA Flash -----------------------------"


for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash via SDPA
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_kv 1 --use_flash2_decode 0 --enable_sdpa_flash 1 
    done
done

echo "------------- Done with SDPA Flash + Non contiguous -----------------------------"



echo "---------------- GQA Aware -----------------"


for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # Bifurcated + compile
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 1 --bifurcated_attn 1 --compile 
    done
done

echo "------------- Done with bifurcated compile -----------------------------"


for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # bifurcated
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 1 --bifurcated_attn 1 
        
    done
done

echo "------------- Done with bifurcated normal -----------------------------"

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash2
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 1 --bifurcated_attn 0 --use_flash2_decode 1 
        
    done
done

echo "------------- Done with flash 2 contiguous mode -----------------------------"

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # mock + flash2
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 1 --bifurcated_kv 1 --use_flash2_decode 1 
        
    done
done

echo "------------- Done with flash 2 non contiguous mode -----------------------------"
