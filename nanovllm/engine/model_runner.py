import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.supported_models import model_dict
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        # event 由 LLMEngine 给定参数
        self.event = event
        # 初始化通信，供model的VocalEmbedding、Liner等模块使用
        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # specific the target causalLM based on the 'model_type' config
        self.model = model_dict[hf_config.model_type](hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        shm_name = "nanovllm"
        if self.world_size > 1: # 多卡
            if rank == 0:
                # 主进程先创建SharedMemory，再 barrier
                try:
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**20)
                    print(f"[rank{rank}] successfully created ShareMemory: {shm_name}")
                except FileExistsError:
                    old_shm = SharedMemory(name=shm_name, create=False)
                    old_shm.unlink()
                    old_shm.close()
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**20)
                    print(f"[rank{rank}] successfully cleared and recreated ShareMemory: {shm_name}")
                except Exception as e:
                    print(f"[rank{rank}] Error in creating ShareMemory: {shm_name}")
                    self.is_running = False
                    dist.destroy_process_group()
                    return
                dist.barrier() 
            else:
                # 子进程等待主进程创建完毕SharedMemory后，跟主进程同步
                # 同步完，进入循环
                dist.barrier()
                try:
                    self.shm = SharedMemory(name=shm_name)
                    print(f"[rank{rank}] successfully connected to ShareMemory: {shm_name}")
                    self.loop()
                except Exception as e:
                    print(f"[rank{rank}] Failed to connected to ShareMemory: {shm_name}")
                    self.is_running = False
    
    def exit(self):
        # 退出函数（析构）
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        # 子进程是自己的 event, 清除标志位
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        # 主进程的 event 是所有子进程组成的 list
        for event in self.event:
            # 类似于 notify()
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        # torch 内存重置
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        # 计算两个约束中较紧的那个
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 只跑 prefill阶段，prefill = True
        self.run(seqs, True)
        # 清空 cache
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        '''
            分配给KV cache显存 = gpu总显存 - 不使用KV cache做1次推理时的显存占用（包括模型本身和推理过程中的中间数据）
            warmup_model 是为了估计到 中间activations的大小 ？
        '''
        config = self.config
        hf_config = config.hf_config
        # total 是物理硬件决定
        # free 是当前 free 的
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 此处 self.world_size 就等于 TPS_size
        # num_kv_heads 等于分配到每个TP 的头数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        # 每个 block 的大小（bytes）：2*num_layers*block_size*num_kv_heads*head_dim*dtype
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads # 在这个TP 实例上的tp head_num
            * head_dim
            * hf_config.torch_dtype.itemsize
        )
        # 最多可以分配出多少个 kv_cache block
        # total * config.gpu_memory_utilization: 计算 GPU允许用于本次任务（模型 + KV Cache）的最大显存上限（字节数）
        # - used（扣除当前已占用显存）
        # peak - current: 表示「本进程历史峰值显存」与「当前实时显存」的差值，对应模型 warmup（一次推理）后，释放的中间数据显存余量(即中间激活值的占用)
        # peak    = 模型权重 + 其他（框架等固定开销，peak和current一样）+ 中间计算得到的临时KV激活 + 除KV以外的其他中间激活值（包括Q、MLP、mask等在内）
        # current = 模型权重 + 其他（框架等固定开销，peak和current一样）
        # -(peak - current）: 表示从扣去模型权重以后的显存里再除掉 「下次推理必须复用的临时中间激活值显存」
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - (peak - current))
            // block_bytes
            # // 向下取整
        )
        assert config.num_kvcache_blocks > 0, "your num_kvcache_blocks is set to {num_kvcache_blocks}"
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            # 将 kv_cache 和实际的tensor对应，module.k_cache/v_cache 见 nanovllm/layers/attention.py 的Attention类
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # k 的 indx 为 0
                module.k_cache = self.kv_cache[0, layer_id]
                # v 的 index 为 1
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        ''' used in warmup and prefill stage 
            批量收集多个序列的 "未缓存 token", 准备完整的计算元数据，一次性完成非缓存 token 的 K/V 计算与 KV Cache 初始化 / append
        '''
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # 0: num_cached_tokens 的已经缓存，无须计算，只处理未缓存的部分
            # extend 可以将一个list append 到 一个list的尾部
            input_ids.extend(seq[seq.num_cached_tokens :])
            # positions 由 seq.num_cached_tokens to seqlen-1，是sep中绝对位置索引
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # 为什么 q只缓存一部分，k要缓存全部？
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            # cumulative sequence lengths 累计序列长度， 用于区分批量中不同序列的边界。
            # 批量中, 序列 1 长度 100、序列 2 长度 200, cu_seqlens_q就是[0, 100, 300]
            # 为后续 FlashAttention 批量计算提供 “序列边界信息”，无需对批量序列进行padding（填充）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup, no block_table, no kv cache
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        ''' prepare for flash-attention ?
            input_ids: 最近生成的 token 的 id
            positions: 最近生成的 token 在其seq中的位置
        '''
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # decode 只算一个 token
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # decode阶段 且 cuda_graph 且 batch_size<512
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        # 采样使用主进程来做
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # logits 是概率分布，softmax在 sampler 里做
        logits = self.run_model(input_ids, positions, is_prefill)
        # sampler 只在 rank0 计算
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # cuda_graph 限制序列数，不需要捕获那么多图
        max_bs = min(self.config.max_num_seqs, 512)
        # 限制单sequence的block的个数，向上取整
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            # 定义 cuda_graph
            graph = torch.cuda.CUDAGraph()
            # is_prefill = False, decode阶段开启cuda_graph
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            # 逆序遍历，第一个graph创建最大的显存池，后续bs的显存占用小于这个bs，可以复用
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
