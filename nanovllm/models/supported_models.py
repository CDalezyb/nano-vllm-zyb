from .qwen3 import Qwen3ForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM

model_dict = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}