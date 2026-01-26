from .qwen3 import Qwen3ForCausalLM
from .qwen2 import Qwen2ForCausalLM

model_dict = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
}