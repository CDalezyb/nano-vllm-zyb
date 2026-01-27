import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
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
>>>>>>> 1b28869 (update bench and example)
