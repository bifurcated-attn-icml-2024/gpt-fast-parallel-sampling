# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import copy
import wandb
import json
import os

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

from model import Transformer


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1) # exponential distribution
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # [b, top_k]
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits, temperature, top_k)    
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, parallel_samples=1, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    ## output: [B, S, V] -> [B, 1, V]
    logits = model(x, input_pos, prefill=True)[:, -1, :] # only the last token
    if parallel_samples > 1:
        logits = logits.repeat(parallel_samples, 1)[:]
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos, prefill=False)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, parallel_samples, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    print("Enabling SDPA flash", args.enable_sdpa_flash)
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=bool(args.enable_sdpa_flash),
                                            enable_mem_efficient=bool(args.enable_mem_efficient),
                                            enable_math=bool(args.enable_math)):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            cur_token = next_token.view(parallel_samples, -1)
            new_tokens.append(cur_token.clone())
            callback(new_tokens[-1]) # what does this do?
            new_probs.append(next_prob.clone())

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


def prepare_parallel_sampling_kv(model, num_parallel_samples, bifurcated_kv=False):
    b = num_parallel_samples
    # expand by reference K and V
    for idx, _ in enumerate(model.layers):
        # for bifurcated, we do not need to repeat the batch indices
        if not bifurcated_kv:
            # expand by reference K and V does not work because K and V include incremental decoding positions
            # another way is to reference only the context part, then make the incremental decoding part is separate
            # [context ref] + [inc1]
            # [context ref] + [inc2]
            # would this work for reference or would it just copy the tensor anyways?
            k_cache = model.layers[idx].attention.kv_cache.k_cache # [1, g, m, k]
            v_cache = model.layers[idx].attention.kv_cache.v_cache # [1, g, m, v]
            model.layers[idx].attention.kv_cache.k_cache = k_cache.repeat(b, 1, 1, 1)
            model.layers[idx].attention.kv_cache.v_cache = v_cache.repeat(b, 1, 1, 1)


def prepare_parallel_sampling(seq, num_parallel_samples, model, bifurcated_kv=False):
    if num_parallel_samples > 1:
        seq = seq.repeat(num_parallel_samples, 1)
        prepare_parallel_sampling_kv(model, num_parallel_samples, bifurcated_kv)
    return seq

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    parallel_samples: int = 1,
    bifurcated_kv: bool = False,
    bifurcated_attn: bool = False,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length, hard_reset=True,
                           bifurcated_kv=bifurcated_kv,
                           bifurcated_attn=bifurcated_attn,
                           context_seq_len=T, max_new_tokens=max_new_tokens,
                           parallel_samples=parallel_samples,
                           use_flash2_prefill=bool(args.use_flash2_prefill),
                           use_flash2_decode=bool(args.use_flash2_decode),
                           use_sdpa_flash=bool(args.enable_sdpa_flash),
                           )
                           
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)
    t0 = time.perf_counter()
    next_token = prefill(model, prompt.view(1, -1), input_pos, parallel_samples, **sampling_kwargs)
    
    if is_speculative:
        prefill(draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq = prepare_parallel_sampling(seq, parallel_samples, model, bifurcated_kv)


    if parallel_samples > 1:
        seq[:, T] = next_token.squeeze() # [b, 1]
    else:
        # seq is of length 'context_len' + 'max_new_tokens'
        seq[T] = next_token
    prefil_time = time.perf_counter() - t0

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)
    
    # model.layers[0].attention.kv_cache.k_cache.size() = bgm'k where m' is the rounded up context length + max new tokens
    # model.layers[0].attention.kv_cache.v_cache.size() = bgm'v

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        next_token_view = next_token.view(args.parallel_samples, -1)
        generated_tokens, _ = decode_n_tokens(model, next_token_view, input_pos, max_new_tokens - 1, parallel_samples, callback=callback, **sampling_kwargs)
        if args.parallel_samples > 1:
            print("shape of generated tokens", generated_tokens[0].shape)
            assert len(generated_tokens[0].shape) == 2
            seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)
        else:
            seq[T + 1:] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {
        'accept_counts': accept_counts,
        'prefill_time': prefil_time
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, max_length=128, bos=True, device='cuda'):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    tokens = tokens[:max_length]
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name,
                                      gqa_aware=args.gqa_aware,
                                      block_size=args.block_size,
                                      )

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-3].startswith("g")
        assert path_comps[-2] in device, "weight packed format mismatch, please rerun quantize.py!"
        groupsize = int(path_comps[-3][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime(use_cuda)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: str = "Hello, my name is",
    prompt_len: int = 128,
    interactive: bool = False,
    num_repetitions: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    device='cuda',
    parallel_samples=1,
    bifurcated_kv=False,
    bifurcated_attn=False,
    wandb_group=None,
    log_filename=None,
    gqa_aware=None,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    encoded = encode_tokens(tokenizer, prompt, max_length=prompt_len, bos=True, device=device)
    encoded_copy = copy.deepcopy(encoded)
    prompt_length = encoded.size(0)

    torch.manual_seed(1234)
    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    if compile:
        if is_speculative and use_tp: # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        print("#### Setting torch.compile")
        # torch.compile will compile `decode_one_token` function to be a fused kernel
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {
        'prefill_time': [],
        'per_step_time': [],
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if compile else 0
    if args.burn_in:
        start -= 1
    # need to burn -- othertime the prefill time for the first one is very high
    
    if wandb_group is not None:
        wandb.init(project='bifurcated_attn',
                group=wandb_group)
        wandb.config.update(
            {
                "prompt_len": prompt_len,
                "interactive": interactive,
                "num_repetitions": num_repetitions,
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "temperature": temperature,
                "checkpoint_path": checkpoint_path,
                "compile": compile,
                "compile_prefill": compile_prefill,
                "profile": profile,
                "speculate_k": speculate_k,
                "device": device,
                "parallel_samples": parallel_samples,
                "bifurcated_attn": bifurcated_attn,
                "wandb_group": wandb_group,
                "log_filename": log_filename,
                "gqa_aware": gqa_aware,
                "device": device,
                "log_filename": log_filename,
                "interactive": interactive,
            }
        )
    
    for i in range(start, num_repetitions):
        device_sync(device=device) # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = copy.deepcopy(encoded)
            # encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_repetitions - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                parallel_samples,
                bifurcated_kv=bifurcated_kv,
                bifurcated_attn=bifurcated_attn,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        if compile and i == start:
            # so the first step essentially lets the model runs and it will auto compile?
            compilation_time = time.perf_counter() - t0
            print(f"Compilation time: {compilation_time:.2f} seconds")
            continue
        if args.burn_in and ((compile and i == start + 1) or (not compile and i == start)):
            burnin_time = 1000*metrics['prefill_time']
            print("Burn in: prefile time = ", 1000*metrics['prefill_time'])
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device) # MKG
        t = time.perf_counter() - t0


        if bool(args.display) and not interactive and i == num_repetitions - 1:
            if parallel_samples > 1:
                if args.display_prompt:
                    print("********** prompt **********")
                    print(tokenizer.decode(encoded.tolist()))
                for j in range(parallel_samples):
                    print(f"---------- Parallel Sample {i}/{j} ---------")
                    print(tokenizer.decode(y[j].tolist()[prompt_length:]))
            else:
                if args.display_prompt:
                    print("********** prompt **********")
                    print(tokenizer.decode(encoded.tolist()))
                print("Generated Text:")
                print(tokenizer.decode(y.tolist()[prompt_length:]))
        else:
            print()
        tokens_generated = y.size(-1) - prompt_length
        tokens_sec = tokens_generated / t
        per_step_time = (t - metrics['prefill_time']) / tokens_generated
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        aggregate_metrics['per_step_time'].append(per_step_time)
        aggregate_metrics['prefill_time'].append(metrics['prefill_time'])
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Prompt length: {prompt_length} | Model block size {model.config.block_size}")
    print(f"Num parallel samples: {parallel_samples}")
    
    print(f"Compile = ** {compile} ** Compile prefill = {compile_prefill} Profile = {profile}")
    print(f"Enable SDPA Flash = {args.enable_sdpa_flash} Enable Mem Efficient = {args.enable_mem_efficient} Enable Math = {args.enable_math}")
    print(f"Bifurcated Attention = ** {bifurcated_attn} Bifurcated KV = ** {bifurcated_kv} ")
    print(f"Using Flash2 prefill = {args.use_flash2_prefill} Using Flash2 decode = {args.use_flash2_decode}")
    print(f"GQA aware = {args.gqa_aware} Quantization = {args.model_quantization}")
    print(f"Burn in time (prefill): {burnin_time:.2f} (ms)")
    memory_used = torch.cuda.max_memory_reserved() / 1e9
    print(f"Memory used: {memory_used:.02f} GB")
    tokens_per_sec = torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item()
    print(f"Average tokens/sec: {tokens_per_sec:.2f}")
    prefill_mean = 1000 * torch.mean(torch.tensor(aggregate_metrics['prefill_time'])).item()
    prefill_std = 1000 * torch.std(torch.tensor(aggregate_metrics['prefill_time'])).item()
    print(f"Prefill time \t {prefill_mean:.2f} +- {prefill_std:.2f} (ms)" )
    per_step_time_mean = 1000 * torch.mean(torch.tensor(aggregate_metrics['per_step_time'])).item()
    per_step_time_std = 1000 * torch.std(torch.tensor(aggregate_metrics['per_step_time'])).item()
    print(f"Per step time\t {per_step_time_mean:.3f} +- {per_step_time_std:.3f} (ms)" )
    print("==================================================")

    result_dict = {
        "prompt_len": prompt_len,
        "num_repetitions": num_repetitions,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "temperature": temperature,
        "compile": compile,
        "compile_prefill": compile_prefill,
        "speculate_k": speculate_k,
        "parallel_samples": parallel_samples,
        "bifurcated_kv": bifurcated_kv,
        "bifurcated_attn": bifurcated_attn,
        "gqa_aware": gqa_aware,
        "use_flash2_prefill": args.use_flash2_prefill,
        "use_flash2_decode": args.use_flash2_decode,
        "enable_sdpa_flash": args.enable_sdpa_flash,
        "enable_mem_efficient": args.enable_mem_efficient,
        "enable_math": args.enable_math,
        "tokens_per_sec": tokens_per_sec,
        "prefill_mean": prefill_mean,
        "prefill_std": prefill_std,
        "per_step_time_mean": per_step_time_mean,
        "per_step_time_std": per_step_time_std,
        "burnin_time": burnin_time,
        "acceptance_probs": acceptance_probs if is_speculative else None,
        "memory_used": memory_used, 
        "mean_accepted": sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated) if is_speculative else None,
    }
    if wandb_group is not None:
        wandb.log(result_dict)
        wandb.finish()

    if log_filename:
        with open(log_filename, 'a') as f:
            f.write(json.dumps(result_dict) + '\n')


def load_content(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--prompt_len', type=int, default=128)
    parser.add_argument('--prompt_file', type=str, default='book.txt')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_repetitions', type=int, default=3, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling.')
    #parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/mistralai/Mistral-7B-v0.1/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use')
    parser.add_argument('--attention_type', type=str, default='sdpa')
    parser.add_argument('--enable_sdpa_flash', type=int, default=0)
    parser.add_argument('--enable_mem_efficient', type=int, default=0)
    parser.add_argument('--block_size', type=int, default=32768)
    parser.add_argument('--display', type=int, default=0)
    parser.add_argument('--burn_in', type=int, default=1)
    parser.add_argument('--parallel_samples',
                        type=int,
                        default=1,
                        help='The number of parallel samples. Enabled if > 1'
                        )
    parser.add_argument('--bifurcated_kv',
                        type=int,
                        default=0,
                        help="Store KV in bifurcated mode"
                        )
    parser.add_argument('--bifurcated_attn',
                        type=int,
                        default=0,
                        help="Perform bifurcated attention"
                        )
    parser.add_argument('--wandb_group', type=str, default=None, help='Wandb group name')
    parser.add_argument('--log_file', type=str, default="results.jsonl", help='Log file name')
    parser.add_argument('--display_prompt', type=int, default=0, help="Display prompt")
    parser.add_argument('--gqa_aware', type=int, default=0, help="GQA aware during attention computation")
    parser.add_argument('--use_flash2_prefill', type=int, default=0, help="Using flash attention for prefill")
    parser.add_argument('--use_flash2_decode', type=int, default=0, help="Using flash attention for decoding")
    parser.add_argument('--model_quantization', type=str, default=None, help="Model quantization")

    args = parser.parse_args()

    if args.model_quantization is not None and args.model_quantization.lower() != "none":
        args.checkpoint_path = Path(args.checkpoint_path.parent / f"{args.checkpoint_path.stem}_{args.model_quantization}.pth")
        print(f"Quantized model path: {args.checkpoint_path}")
    
    if args.enable_sdpa_flash:
        args.enable_math = 0
    elif args.enable_sdpa_flash == 0:
        args.enable_math = 1

    if args.bifurcated_attn:
        print(f"Setting bifurcated_kv memory to {args.bifurcated_kv} since it's required for bifurcated attention.")
        args.bifurcated_kv = 1

    if args.prompt_file is not None and os.path.exists(args.prompt_file):
        args.prompt = load_content(args.prompt_file)
    main(
        args.prompt, args.prompt_len, args.interactive, args.num_repetitions, args.max_new_tokens, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.profile, args.draft_checkpoint_path,
        args.speculate_k, args.device, args.parallel_samples, bool(args.bifurcated_kv), bool(args.bifurcated_attn),
        args.wandb_group, args.log_file, args.gqa_aware
    )
