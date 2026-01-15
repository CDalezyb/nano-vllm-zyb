from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

# 关于 _CONTEXT 的访问和赋值规则：
# 如果是仅访问：函数内部试图访问一个变量时，解释器会先在「局部作用域」（函数内部）查找该变量；
#              如果局部作用域中不存在，会自动向上查找「全局作用域」（模块级别）的变量；
#               因此get_context()可以直接获取全局的_CONTEXT，无需显式声明global。
# 如果需要修改：如果不加 global，python会认为_CONTEXT 是函数内的局部变量，会创建局部变量，全局作用域的_CONTEXT没有改变
_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
