#PROMPT_LENGTHS=(8192 ) 
PROMPT_LENGTHS=(16384 ) 
PARALLEL_SAMPLES=(1 2 4 8 16 32 64 128 256 512 1024 2048)

export GROUP_NAME="bifurcated_results"


for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # SDPA
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 0 --use_flash2_decode 0 

    done
done



for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 0 --use_flash2_decode 1
        
    done
done



for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 1 --use_flash2_decode 0
        
    done
done


for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 1 --bifurcated_attn 0 --use_flash2_decode 1
        
    done
done


for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 1 --bifurcated_attn 1 --use_flash2_decode 0
        
    done
done


for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # SDPA + compile
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 0 --use_flash2_decode 0 --compile 
    done
done

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # Bifurcated + MHA
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 0 --bifurcated_attn 1 --use_flash2_decode 0 --compile 
    done
done

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for parallel_sample in "${PARALLEL_SAMPLES[@]}"; do
        # Bifurcated + GQA
        python generate.py --prompt_len $prompt_len --parallel_samples $parallel_sample \
        --use_flash2_prefill 1 \
         --gqa_aware 1 --bifurcated_attn 1 --use_flash2_decode 0 --compile 
    done
done
