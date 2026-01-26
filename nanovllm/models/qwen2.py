import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen2Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # QKVParallelLinear 是 ColumnParallel
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True, # why True here, the config file has no bias attribution
            # Qwen2系列默认QKV运算有bias，从Qwen3移除了，在QK以后添加了norm，获得数值稳定性
        )
        
        # o需要规约（all_reduce)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
       
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        ''' 
            in prefill phase: S = prompt_length
            in decode  phase: S = 1
        '''
        # hidden_states: [S, hidden_size]
        # qkv :[ S, q_size + 2 * kv_size]
        # q_size = num_heads  * head_dim
        # kv_size = num_kv_heads  * head_dim
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q :[ S,    num_heads, head_dim]
        # kv :[ S, num_kv_heads, head_dim], num_heads%num_kv_heads = 0
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # decode 时，positions 是最近生成的 token 在 seq 中的绝对位置（位置计数包括prompt）
        # prefill时：positions 是 prompt 的所有 token 的位置
        q, k = self.rotary_emb(positions, q, k)
        # Step1： attention 之前的shape
        #   q shape       : [S, num_heads, head_dim] ,S = S_in if prefill else 1
        #   kv_cache shape: [S_in+S_out+1, num_kv_heads, head_dim]
        # Step2： 将Q和KV头数进行对齐，GQA，将kv复制若干次（逻辑层上复制，其实物理层上没有复制，因为数据是相同的）
        #   q shape       : [S, num_heads, head_dim] ,S = S_in if prefill else 1
        #   kv_cache shape: [S_in + S_out + 1, num_heads, head_dim]
        # Step3：SDPA计算：softmax(Q@K^T)
        #   Q reshape   :  [               1, num_heads, head_dim] to [num_heads,                1, head_dim]
        #   K.reshape^T :  [S_in + S_out + 1, num_heads, head_dim] to [num_heads, head_dim, S_in + S_out + 1]
        #   Q@K^T shape :  [num_heads, 1, S_in + S_out + 1]
        #   softmax 对 sql_len维度做，不改变维度
        # Step-4: atten_scores@V
        #   V.reshape     :  [S_in + S_out + 1, num_heads, head_dim] to [num_heads, S_in + S_out + 1, head_dim]
        #   atten_scores@V:  [num_heads, 1, S_in + S_out + 1] @ [num_heads, S_in + S_out + 1, head_dim] = [num_heads, 1, head_dim]
        # 最后会做 reshape :  [1, num_heads, head_dim]
        # o shape:  [S, self.num_heads, self.head_dim](decode, S=1)
        o = self.attn(q, k, v)
        # o.flatten(1, -1)： 从idx=1 的维度进行 flatten
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        '''
            gate_up_proj  ColumnParallelLinear(TP)
            down_proj     RowParallelLinear(TP)
        '''
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None: # 第一层 layer，对hidden_states 做norm，并将其作为残差
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else: # 后续 layer
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # attention 计算与残差更新
        hidden_states = self.self_attn(positions, hidden_states)
        # post_attention_layernorm 表示在 attention 后做的一个 PRE-NORM！！！
        # 此处，返回的 residual 其实就是 输入的 hidden_states
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # embedding first
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            # Qwen2DecoderLayer
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后跟着一个 RMSNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    ''' ForCausalLM 表示是一个因果模型，有causal mask'''
    # 用于将模型权重读取，由于有些权重进行合并
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings: # Qwen2, tie_word_embeddings is False
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
