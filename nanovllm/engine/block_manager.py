from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    ''' This is the class of block:  nanovllm 的最小物理块单元, 对应KV Cache的一个物理块, 存储连续token序列 '''

    def __init__(self, block_id):
        self.block_id = block_id   # 当前块的唯一ID，全局唯一，初始化时由 BlockManager 分配
        self.ref_count = 0         # 【核心】块的引用计数，实现多序列共享复用块的关键，0=空闲，>0=被引用
        self.hash = -1             #  # 当前块的哈希值，由compute_hash生成，-1=未计算/无效哈希
        # hash 值不会为 -1 吗？
        self.token_ids = []        # tokens are stored in this block

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    '''负责block 的分配、管理和销毁:  KV Cache物理块的全生命周期管家
       核心能力：分配块、释放块、复用缓存块、追加token扩容块、引用计数管理
    '''

    def __init__(self, num_blocks: int, block_size: int):                  # num_blocks表示可用的block数，由 scheduler 给定
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]   # 全局所有物理块的数组，索引=block_id，快速通过id取块
        self.hash_to_block_id: dict[int, int] = dict()                     # 哈希值 → 块ID的映射表【缓存复用核心】，通过哈希快速找复用块
        self.free_block_ids: deque[int] = deque(range(num_blocks))         # 空闲的 blocks 的 ids
        self.used_block_ids: set[int] = set()                              # 在使用的 block的ids

    @classmethod
    # 根据 token_ids 和 prefix 计算hash值，作为这个token_ids的索引
    # 只对存满的 block 计算 hash_value
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """[私有方法]: 分配指定ID的块，内部调用，不对外开放
        行为：
            1. 将 block_id 从 free_block_ids 双向队列移除
            2. 将 block_id 加入 used_block_ids set中
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> None: # ？应该没有返回值才对
        '''[私有方法]: 释放某个id的block，内部调用，引用计数=0时触发'''
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        # ？ 为什么不清理 hash_to_block_id 对应的 block_id 值

    def can_allocate(self, seq: Sequence) -> bool:
        """判断：是否有足够的空闲块，为指定序列分配所需的全部块，prefill时使用"""
        return len(self.free_block_ids) >= seq.num_blocks
    
    def can_append(self, seq: Sequence) -> bool:
        """判断：是否有足够的空闲块，为指定序列分配下一个块，decode时使用"""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # allocate kv_cache_blocks for the given Sequence in prefill stage
    def allocate(self, seq: Sequence):
        assert not seq.block_table, "分配块时，序列的块表必须为空"
        h = -1              # 临时哈希值，用于链式计算块哈希
        cache_miss = False  # 缓存失效标记：True=无复用块，需要分配新块；False=命中缓存，复用块
        for i in range(seq.num_blocks):
            # token_ids = {当前 block的 block_size 个tokens}, 计算block_idx并添加到seq的 block_table 里
            token_ids = seq.block(i)
            # 只有当token数量等于块容量时，才计算哈希（完整块才缓存），否则哈希=-1
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            # 查哈希表，（包括前缀的哈希值）看是否有已缓存的块能复用，没有则返回 -1
            block_id = self.hash_to_block_id.get(h, -1)
            # 缓存失效条件：无哈希对应块 或 块的token内容不一致（哈希碰撞兜底校验）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            # 从某个block开始cache miss了，后续的块都会miss
            # 分支1：缓存失效 → 分配新块（从空闲队列取第一个）
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            # 分支2：缓存命中 → 复用已有块，prefix cache 优化点！
            else:
                seq.num_cached_tokens += self.block_size
                # 复用的块如果已在用，只需要引用计数+1即可，不用重新分配
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                # 复用的块如果是空闲状态，重新分配
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            # 将分配的全局块ID，加入序列的块表，完成映射
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """核心方法：释放指定序列占用的所有块（序列推理结束/被驱逐时调用）"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def may_append(self, seq: Sequence):
        """核心方法：为序列追加新token时，扩容/更新块（decode时token流式生成的核心逻辑），in decode stage"""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # 情况1：序列长度%块容量=1 → 最后一个块已满, 现在生成了一个新的token, 需要分配新块来存储
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # 情况2：序列长度%块容量=0 → 最后一个块刚存满，计算哈希并加入缓存表
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix_hash = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix_hash)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        # 情况3：序列长度%块容量≠0/1 → 最后一个块还没满，无需做处理，哈希保持-1
        else:
            assert last_block.hash == -1
