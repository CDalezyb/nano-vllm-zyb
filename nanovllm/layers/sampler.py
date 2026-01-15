import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 广播，logits.shape：[bs, vocab_size], temperatures.shape:[bs]
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # probs: [bs, vocab_size]
        probs = torch.softmax(logits, dim=-1)
        # argmax 操作之前都是 [bs, vocal_size], 之后变成 [bs]
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
