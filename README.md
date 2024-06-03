# Bifurcated Attention Results


## Summary
- Below, we show that context-aware bifurcated attention helps reduce parallel sampling latency significantly for both MHA and GQA architectures.
- Bifurcated attention is implemented in native torch, which can directly benefit from torch compile and outperforms FlashAttention2.
- We note that the bifurcated attention kernel is only for the `decode` step, which means that in the `prefill` step, we can use any kernel that is efficient such as `flash`. For example, with context length `8192`, FlashAttention2 results in latency ~ `130 ms` compared to `247 ms` for Torch SDPA. That is, we can use any kernel at prefill phase and use bifurcated attention for high decoding workload.
- For non context aware kernel, storing all KV in contiguous memory incurs significant memory cost for parallel sampling. In order to avoid out-of-memory,
we also include the `non contiguous` setting we use the bifurcated attention memory setup (keeping only one copy of the context and expand by reference to different batch indices). In the contiguous memory case, we keep explicit KV cache of context for all batch indices. We show that even though the non contiguous case avoids early OOM, the latencies are still much higher than bifurcated attention.
- Native flash 2 is not yet compatible with torch compile.


## Comparing kernels

We show different workloads with various number of `parallel samples` where include results for both MHA and GQA. Results are obtained with 1 H100 GPU for 7B model.

### MHA

- MHA is more IO intensive than GQA, therefore bifurcated attention helps significantly even compared to highly efficient kernels. Below, we use context length 8K with varying number of parallel samples.


#### MHA 8K Context

| # Parallel Samples | Bifurcated + Compile | Bifurcated | Flash2 | Torch SDPA Math | Torch SDPA Math + compile | Flash2 + Non contiguous | SDPA Flash | SDPA Flash + Non contiguous | SDPA Math + Non contiguous + Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.639 | 30.389 | 24.069 | 26.397 | **8.776** | 24.543 | 22.00 | 23.431 | 10.665 |
| 2 | 11.774 | 31.371 | 24.498 | 28.706 | **10.505** | 31.533 | 24.771 | 31.658 | 14.449 |
| 4 | **12.030** | 31.440 | 39.664 | 43.361 | 13.227 | 50.539 | 38.867 | 51.065 | 23.205 |
| 8 | **12.358** | 33.722 | 60.924 | 72.705 | 17.330 | 84.519 | 61.225 | 84.987 | 35.421 |
| 16 | **12.595** | 31.707 | 109.647 | 132.89 | 26.192 | 155.847 | 109.457 | 159.816 | 63.679 |
| 32 | **13.471** | 31.788 | 205.578 | 251.024 | - | 305.395 | 205.921 | 306.598 | 120.395 |
| 64 | **15.355** | 35.267 | OOM | OOM | - | 599.084 | - | 601.480 | 238.192 |
| 128 | **19.561** | 48.699 | - | - | - | 1183.460 | - | OOM | OOM |
| 256 | **27.146** | 75.212 | - | - | - | 1842.982 | - | - | - |
| 512 | **44.332** | 130.587 | - | - | - | - | - | - | - |
| 1024 | **81.146** | 242.738 | - | - | - | - | - | - | - |
| 2048 | OOM | 473.741 | - | - | - | - | - | - | - |

#### MHA 16K Context

| # Parallel Samples | Bifurcated + Compile | Bifurcated | Flash2 | Torch SDPA Math | Torch SDPA Math + compile | Flash2 + Non contiguous | SDPA Flash | SDPA Flash + Non contiguous | SDPA Math + Non contiguous + Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12.163 | 30.658 | 26.282 | 30.134 | 13.055 | 30.485 | 26.224 | 30.195 | 15.525 |
| 2 | 17.170 | 32.619 | 37.723 | 44.737 | **15.349** | 51.304 | 38.248 | 51.235 | 22.456 |
| 4 | **17.327** | 33.438 | 65.980 | 73.621 | 20.650 | 91.245 | 65.828 | 90.755 | 39.511 |
| 8 | **18.070** | 34.665 | 110.313 | 132.294 | 32.058 | 159.959 | 110.552 | 160.391 | 64.221 |
| 16 | **18.462** | 36.780 | 206.926 | 251.473 | OOM | 306.745 | 206.517 | 307.313 | 119.871 |
| 32 | **19.920** | 41.927 | OOM | OOM | - | 601.096 | OOM | 603.612 | 237.891 |
| 64 | **22.958** | 50.530 | - | - | - | 1195.347 | - | OOM | OOM |
| 128 | **28.976** | 68.306 | - | - | - | 1908.226 | - | - | - |
| 256 | **40.070** | 106.100 | - | - | - | OOM | - | - | - |
| 512 | **65.020** | 183.143 | - | - | - | - | - | - | - |
| 1024 | **117.753** | 339.738 | - | - | - | - | - | - | - |
| 2048 | OOM | 660.198 | - | - | - | - | - | - | - |


#### MHA 32K Context

| # Parallel Samples | Bifurcated + Compile | Bifurcated | Flash2 | Torch SDPA Math | Torch SDPA Math + compile | Flash2 + Non contiguous | SDPA Flash | SDPA Flash + Non contiguous | SDPA Math + Non contiguous + Compile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20.898 | 39.972 | 37.674 | 44.942 | **19.797** | 67.443 | 37.463 | 67.299 | 30.394 |
| 2 | **29.338** | 48.614 | 55.941 | 69.224 | OOM | 156.610 | 55.855 | 156.352 | 47.625 |
| 4 | **29.726** | 49.768 | OOM | OOM | - | 300.468 | OOM | 300.965 | 90.191 |
| 8 | **30.295** | 51.309 | - | - | - | 567.933 | - | 568.811 | 152.187 |
| 16 | **30.657** | 54.921 | - | - | - | 670.205 | - | 672.421 | 290.593 |
| 32 | **32.149** | 62.284 | - | - | - | 1318.045 | - | 1323.246 | 569.741 |
| 64 | **35.254** | 75.220 | - | - | - | OOM | - | OOM | OOM |
| 128 | **41.440** | 101.175 | - | - | - | - | - | - | - |
| 256 | OOM | 159.089 | - | - | - | - | - | - | - |
| 512 | - | 277.047 | - | - | - | - | - | - | - |
| 1024 | - | OOM | - | - | - | - | - | - | - |



### GQA

- For GQA, bifurcated attention is able to help scale to very large inference workload. Using torch.compile mode makes the inference much faster compared than Flash2. Below, we consider context length 8K, 16K and 32K.
- Note that torch SDPA does not directly support and the latency will be the same as the MHA case (which is worse, and is not included for direct comparison here).



#### GQA with 8K context length

| # Parallel Samples | Bifurcated + Compile | Bifurcated | Flash2 | Flash2 (non contiguous) |
| --- | --- | --- | --- | --- |
| 1 | **10.561** | 28.365 | 21.760 | 23.475 |
| 2 | **11.351** | 29.526 | 22.460 | 39.930 |
| 4 | **11.515** | 29.578 | 22.570 | 71.567 |
| 8 | **11.786** | 29.576 | 22.649 | 126.353 |
| 16 | **11.719** | 30.265 | 22.310 | 240.963 |
| 32 | **12.495** | 29.755 | 26.061 | 468.934 |
| 64 | **13.866** | 29.515 | OOM | 403.078 |
| 128 | **17.033** | 29.547 |  | 788.658 |
| 256 | **24.381** | 40.070 |  | ?? |
| 512 | **39.080** | 65.737 |  | ?? |
| 1024 | **72.238** | 118.572 |  |  |
| 2048 | OOM | 230.879 |  |  |



#### GQA with 16K context length

| # Parallel Samples | Bifurcated + Compile | Bifurcated | Flash2 | Flash2 (non contiguous) |
| --- | --- | --- | --- | --- |
| 1 | **15.164** | 30.966 | 23.588 | 25.226 |
| 2 | **15.985** | 32.155 | 23.782 | 28.531 |
| 4 | **16.202** | 32.188 | 24.218 | 42.469 |
| 8 | **16.612** | 32.406 | 24.029 | 70.009 |
| 16 | **16.682** | 32.846 | 30.194 | 130.772 |
| 32 | **17.772** | 32.747 | OOM | 244.543 |
| 64 | **19.900** | 32.067 |  | 482.713 |
| 128 | **24.899** | 40.258 |  | 465.696 |
| 256 | **33.760** | 59.418 |  | 915.892 |
| 512 | OOM | OOM |  | OOM |
| 1024 |  |  |  |  |
| 2048 |  |  |  |  |



#### GQA with 32 K context length

| # Parallel Samples | Bifurcated + Compile | Bifurcated | Flash2 | Flash2 (non contiguous) |
| --- | --- | --- | --- | --- |
| 1 | **22.786** | 37.204 | 26.635 | 28.197 |
| 2 | **23.722** | 37.469 | 26.815 | 45.704 |
| 4 | **23.980** | 37.481 | 27.304 | 72.938 |
| 8 | **24.586** | 38.120 | 28.355 | 127.956 |
| 16 | **24.868** | 37.291 | OOM | 245.808 |
| 32 | **27.005** | 37.844 |  | 467.610 |
| 64 | **30.313** | 45.731 |  | 463.546 |
| 128 | **37.601** | 63.055 |  | 909.020 |
| 256 | **52.064** | 96.277 |  | 1805.599 |
| 512 | OOM | OOM |  | OOM |
| 1024 |  |  |  |  |
| 2048 |  |  |  |  |



## Applicability with Higher Tensor Parallelism and Model Quantization

We show that our method works out of the box together with other inference techniques such as tensor parallelism (to decrease latency and memory consumption) and model quantization (lower memory consumption). 


### Model Quantization with Int8

- Context length 8192 and num parallel samples = 8
- Note: int8 quantization results in lower memory usage but is slightly slower due to the int8 to floating point conversion which can cause additional IO. However, in each setting, either `int8` or `bf16`, our method is able to improve the latency compared to torch SDPA and FlashAttention2.

|           | Bifurcated | SDPA | Flash2 | Bifurcated + Compile | SDPA + Compile | Flash2 + Compile |
|-----------|------------|------|--------|----------------------|----------------|------------------|
| int8      | 44.328     | 92.720 | 76.609 | 21.817               | 24.391         | N/A              |
| bf16      | 31.332     | 72.637 | 56.363 | 14.394               | OOM            | N/A              |


### TP=2 with Mistral 7B (8K context, 8 parallel samples)

- Higher tensor parallelism is usually required to higher inference workload. Our method works out of the box without additional modification for tensor parallelism.

| Context Length | Batch size | SDPA    | Bifurcated | Flash2   |
| -------------- | ---------- | ------- | ---------- | -------- |
| 16384          | 16         | 131.460 | 55.515     | 92.115   |
| 32640          | 8          | 133.851 | 58.555     | 92.354   |
| 32640          | 16         | 246.531 | 57.995     | 162.016  |
| 32640          | 32         | OOM     | 57.861     | OOM      |
| 32640          | 64         |         | 60.325     |          |
| 32640          | 128        |         | 67.823     |          |





<br>
<br>
<br>
<br>

Original README for gpt-fast is retained below.

# gpt-fast
Simple and efficient pytorch-native transformer text generation.

Featuring:
1. Very low latency
2. <1000 lines of python
3. No dependencies other than PyTorch and sentencepiece
4. int8/int4 quantization
5. Speculative decoding
6. Tensor parallelism
7. Supports Nvidia and AMD GPUs

This is *NOT* intended to be a "framework" or "library" - it is intended to show off what kind of performance you can get with native PyTorch :) Please copy-paste and fork as you desire.

For an in-depth walkthrough of what's in this codebase, see this [blog post](https://pytorch.org/blog/accelerating-generative-ai-2/).

## Supported Models

### LLaMA family
Please check the rest of this page about benchmark of LLaMA family models.

### Mixtral 8x7B
We also supported [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) which is a high-quality sparse mixture of experts (MoE) model, the average token generation rates are:

|                  |   1 GPU |    2 GPU  | 4 GPU  |    8 GPU    |
|------------------|---------|-----------|--------|------------|
|baseline(bfloat16)|    OOM  |    78.75  | 118.23 |  203.69    |
|        int8      |   56.04 |    99.91  | 149.53 |  218.48    |

Note that the benchmarks run on an 8xA100-80GB, power limited to 330W with a hybrid cube mesh topology. Note that all benchmarks are run at *batch size=1*, making the reported tokens/s numbers equivalent to "tokens/s/user". In addition, they are run with a very small prompt length (just 5 tokens).

For more details about Mixtral 8x7B, please check [this page](./mixtral-moe) or this [note](https://thonking.substack.com/p/short-supporting-mixtral-in-gpt-fast).

## Community

Projects inspired by gpt-fast in the community:

- [gpt-blazing](https://github.com/armed-gpt/gpt-blazing): applies the same performance optimization strategy to more models (e.g., baichuan2).
- [gptfast](https://github.com/MDK8888/GPTFast): applies a subset of the performance optimizations to all Huggingface models

## Installation
[Download PyTorch nightly](https://pytorch.org/get-started/locally/)
Install sentencepiece and huggingface_hub
```bash
pip install sentencepiece huggingface_hub
```

To download llama models, go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access.
Then login with `huggingface-cli login`



## Downloading Weights
Models tested/supported
```text
openlm-research/open_llama_7b
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-chat-hf
codellama/CodeLlama-7b-Python-hf
codellama/CodeLlama-34b-Python-hf
mistralai/Mistral-7B-v0.1
mistralai/Mistral-7B-Instruct-v0.1
mistralai/Mistral-7B-Instruct-v0.2
```

For example, to convert Llama-2-7b-chat-hf
```bash
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_REPO
```

## Benchmarks
Benchmarks run on an 8xA100-80GB, power limited to 330W with a hybrid cube mesh topology. Note that all benchmarks are run at *batch size=1*, making the reported tokens/s numbers equivalent to "tokens/s/user". In addition, they are run with a very small prompt length (just 5 tokens).

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | Base    |  104.9  | 1397.31 |
|           | 8-bit   | 155.58   | 1069.20 |
|           | 4-bit (G=32)   | 196.80   | 862.69 |
| Llama-2-70B | Base    | OOM     ||
|           | 8-bit   | 19.13    | 1322.58 |
|           | 4-bit (G=32)   | 25.25    | 1097.66 |

### Speculative Sampling
[Verifier: Llama-70B (int4), Draft: Llama-7B (int4)](./scripts/speculate_70B_int4.sh): 48.4 tok/s

### Tensor Parallelism
| Model    | Number of GPUs | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | 1    |  104.9  | 1397.31 |
|           | 2   | 168.84   | 1181.99 |
|           | 4   | 254.02   | 955.83 |
|           | 8   | 328.43   | 704.10 |
| Llama-2-70B  | 1    |  OOM  |  |
|           | 2   | 21.32   | 1481.87 |
|           | 4   | 38.01   | 1340.76 |
|           | 8   | 62.50   | 1135.29 |

### Tensor Parallelism + Quantization
| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-70B | Base    | 62.50     | 1135.29 |
|           | 8-bit   | 80.44    | 752.04 |
|           | 4-bit (G=32)   | 90.77    | 548.10 |

### AMD
Benchmarks run on one GCD of a MI-250x.

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | Base    |  76.33  | 1028.70 |
|           | 8-bit   | 101.86   | 700.06 |

## Generate Text

Model definition in `model.py`, generation code in `generate.py`.

```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```

To squeeze out a little bit more performance, you can also compile the prefill with `--compile_prefill`. This will increase compilation times though.

## Quantization
Choose device to use by
```bash
# The current support devices: cuda, cpu
export DEVICE=cuda
```
### Int8 Weight-Only Quantization
To generate this version of the model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int8.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
```
To run with int8, just pass the int8 checkpoint to generate.py.
```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --device $DEVICE
```

### Int4 Weight-Only Quantization
To generate int4 version of model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4.g32.$DEVICE.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4 --groupsize 32 --device $DEVICE
```

To run with int4, just pass the int4 checkpoint to generate.py.
```bash
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.$DEVICE.pth --compile --device $DEVICE
```

## Speculative Sampling
To generate with speculative sampling (DRAFT_MODEL_REPO should point to a smaller model compared with MODEL_REPO).

In this example, the "smaller" model is just the int8 quantized version of the model.
```
export DRAFT_MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth
```

Note: Running on an A100 80GB, albeit power-limited to 330 watts. Empirically, seems like peak bandwidth is about 1700 GB/s.


## Tensor Parallelism
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=2 generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```

## Experimental
### Evaluation
We use the EleutherAI evaluation harness to evaluate our model accuracy. To evaluate the accuracy, make sure the evaluation harness is installed and pass your model checkpoint and desired tasks to eval.py.

```bash
python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --compile --tasks hellaswag winogrande
```

Note: Generative tasks are currently not supported for gpt-fast

Installation Instructions for the evaluation harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/master#install

### GPTQ
We have a pure pytorch implementation of GPTQ that utilizes torch._dynamo.export to access the model structure. You can generate a GPTQ quantized
version of int4 quantization by using the same command to quantize it but adding 'gptq' to the quantization mode i.e.
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4-gptq.g32.pth
python quantize.py --mode int4-gptq --calibration_tasks wikitext --calibration_seq_length 2048
```

You can then eval or generate text with this model in the same way as above.

## License

`gpt-fast` is released under the [BSD 3](https://github.com/pytorch-labs/gpt-fast/main/LICENSE) license.

## Acknowledgements
Thanks to:
* Lightning AI for supporting pytorch and work in flash attention, int8 quantization, and LoRA fine-tuning.
* GGML for driving forward fast, on device inference of LLMs
* Karpathy for spearheading simple, interpretable and fast LLM implementations
* MLC-LLM for pushing 4-bit quantization performance on heterogeneous hardware
