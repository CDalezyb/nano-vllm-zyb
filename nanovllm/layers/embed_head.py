import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    '''
        从 num_embeddings个 词表 映射到  embedding_dim维度的 token 向量
        是词表并行的 Embedding 层，核心设计目标是将超大的词表（比如百万级）拆分成多个分片，分布在不同的 TP 进程上，从而降低单卡的显存占用。
    '''
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # weight的维度：(num_embeddings_per_partition, embedding_dim)
        # 当TP=1时，num_embeddings_per_partition == num_embeddings
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        '''
            针对有TP设置的 权重加载
        '''
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # x.shape: [batch_size, seq_len]
        # 多个TP节点中存储的weight的显存占用有节省
        # 但是 forward过程中的x、mask和y变量的维度相比于不使用TP没有区别
        if self.tp_size > 1: # if 有 tp
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # mask.unsqueeze(1)： mask from [bs, sql] -> [bs, sql, embedding_dim]
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # cu_seqlens_q是「累积序列长度」张量（比如[0,5,8]表示2个序列，长度5和3）
            # cu_seqlens_q[1:]取每个序列的结束索引，-1得到每个序列最后一个token的索引
            last_indices = context.cu_seqlens_q[1:] - 1
            # 只保留每个序列最后一个token的隐藏状态，contiguous保证内存连续
            x = x[last_indices].contiguous()
        # 步骤3：线性变换生成logits分片（TP下每个rank只输出词表的1/tp_size分片）
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            # 同步通信操作，将所有 rank 的logits分片收集到 rank0的all_logits列表中；
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
