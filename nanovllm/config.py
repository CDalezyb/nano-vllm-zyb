import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model: str                          # Path to the pretrained model directory
    max_num_batched_tokens: int = 16384 # Maximum tokens that can be processed in a single batch
    max_num_seqs: int = 512             # Maximum num of sequences
    max_model_len: int = 4096           # Sequence length the model can handle (1/51)
    gpu_memory_utilization: float = 0.9 # Fraction of GPU memory to allocate (0.9 = 90%)
    tensor_parallel_size: int = 1       # Number of GPUs to use for tensor parallelism (1-8)
    enforce_eager: bool = False         # Whether to disable CUDA graphs and use eager mode
    hf_config: AutoConfig | None = None # Will store HuggingFace model config
    eos: int = -1                       # End-of-sequence token ID (default: -1 means not set)
    kvcache_block_size: int = 256       # Size of blocks for KV cache allocation (must be multiple of 256)
    num_kvcache_blocks: int = -1        # Total number of KV cache blocks (-1 means auto-calculate)

    def __post_init__(self):
        assert os.path.isdir(self.model) # Verify model path is a valid directory
        assert self.kvcache_block_size % 256 == 0 # KV cache block size must be multiple of 256
        assert 1 <= self.tensor_parallel_size <= 8 # Tensor parallelism GPU nums limit 1-8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len # Batch capacity check