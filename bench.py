import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
import argparse

def main(args):
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024
    path = os.path.expanduser(args.model_path)
    print(f"Load model from {path}")

    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )
    
    
if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="nano vllm")
    # default_model_path = os.path.join(os.getenv("MODEL_ROOT_DIR"), "Qwen/Qwen3-0.6B")
    # default_model_path = os.path.join(os.getenv("MODEL_ROOT_DIR"), "Qwen/Qwen2.5-14B")
    default_model_path = os.path.join(os.getenv("MODEL_ROOT_DIR"), "Qwen/Qwen3-30B-A3B")
    argparse.add_argument("--model-path", type=str, default=default_model_path)
    argparse.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    argparse.add_argument("--enforce-eager", type=bool, default=True)
    argparse.add_argument("--temperature", type=float, default=0.6)
    argparse.add_argument("--max-tokens", type=int, default=512)
    args = argparse.parse_args()

    main(args)
