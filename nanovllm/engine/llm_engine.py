import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 获取 Config 的 keys
        config_fields = {field.name for field in fields(Config)}
        # 将 手动设置的参数 赋值给 config_kwargs，以键值对的形式
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 构建Config，没有给定的使用默认值
        config = Config(model, **config_kwargs)
        # 多卡/多节点
        self.ps = [] # 存储子进程（TP1~TPn）的Process对象
        self.events = [] # 存储进程间同步的Event对象
        ctx = mp.get_context("spawn") # 获取multiprocessing的spawn上下文 ? 这是啥
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 主节点（TP0)，主进程
        # TP0的ModelRunner会在初始化时修改config.num_kvcache_blocks, 供后续Scheduler和BlockManager使用
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)                                            # Scheduler
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        ''' 将给定的 prompt 和 sampling_params 组装成 Sequence并且将其加入调度器中 '''
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # nano-vllm 是 prefill 优先, prefill 做完以后再做decode
        seqs, is_prefill = self.scheduler.schedule()
        # 实际上调用 model_runner, 进行推理
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # postprocess 会根据 model_runer 最新生成的 token_id 更新seq，将新生成的token放在seq的token最后面
        # 判断 seq 是否已经生成完毕（达到最大长度或者EOS）, 完毕的就从 scheduler.running 队列中移除并且释放block资源
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 保证 sampling_params 和 prompts 一一对应
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            # outpu代表：一次 prefill/decode step 生成的输出，output = (seq.seq_ids, seq.completion_token_ids)
            # num_tokens代表：此次生成 token 个数？还是
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, generated_token_ids in output:
                outputs[seq_id] = generated_token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
