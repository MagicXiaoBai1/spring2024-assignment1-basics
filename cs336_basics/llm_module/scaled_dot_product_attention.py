from typing import Optional
import torch


def scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:

    raw_attention = Q @ K.transpose(-1, -2)
    raw_attention = raw_attention / (K.size(-1) ** 0.5)
    if mask is not None:
        raw_attention = raw_attention.masked_fill(mask == 1, float("-inf"))
    attention = torch.softmax(raw_attention, dim=-1)
    if pdrop is not None:
        attention = torch.nn.functional.dropout(attention, p=pdrop)
    output = attention @ V
    return output
