# based on file results.jonl
# filter to only wandb_group = args.wandb_group
import json
import argparse
import pandas as pd


def main(args):
    with open("results.jsonl") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    df = df[df["wandb_group"] == args.wandb_group]
    df = df[df["prompt_len"] == args.prompt_len]
    df = df[df["compile"] == bool(args.compile)]
    df = df[df["bifurcated_attn"] == bool(args.bifurcated_attn)]

    # only keep columns parallel_samples, compile 
    df = df[["parallel_samples", "compile", "per_step_time_mean", "bifurcated_attn"]] # , "accuracy", "wandb_group", "prompt_len", "model", "seed", "time"]]

    import pprint
    print(df)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_group", type=str, required=True)
    parser.add_argument("--prompt_len", type=int, required=True)
    parser.add_argument("--compile", type=int, required=True)
    parser.add_argument("--bifurcated_attn", type=int, required=True)
    args = parser.parse_args()
    main(args)


"""
python analyze_result.py --wandb_group "compare_bifurcated_v4" --prompt_len 8192 --compile False --bifurcated_attn False


"""