from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        '''只是确定哪些 Sequence 会被调度到+分配kvcache blocks，并不完成实际运行, 实际运行在 LLMEngine里调用 model_runner.run()'''
        
        # 1. prefill 阶段
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # waiting 队列有等待prefill的 Sequence，并且 num_seqs < 设定的 max_num_seqs
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 为一个prefill阶段的新Sequence 分配 block
            self.block_manager.allocate(seq)
            # seq.num_cached_tokens 在 allocate阶段被设置，如果有prefix cache，则大于0 
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        # 这次 schedule如果有prefill，则不进行decode, 提前返回
        if scheduled_seqs:
            return scheduled_seqs, True

        
        # 2. decode 阶段
        # running 队列有等待decode的Sequence 且 num_seqs < 设定的 max_num_seqs
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # preempt 当前 running队列 最后一个 Sequence，腾出资源给当前 Sequence，当前 Sequence 继续尝试分配直到成功
                    self.preempt(self.running.pop())
                else:
                    # 如果 running 队列已经空了，说明当前 Sequence decode 需要的资源超过了整个系统的 capacity，这时只能放弃这个 Sequence，提前返回
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                # 将输出完成/达到最大输出限制的 Sequence 标记为 FINISHED，并且释放它占用的块资源
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
