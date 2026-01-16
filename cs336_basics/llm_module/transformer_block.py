import torch
import torch.nn as nn
from typing import Optional

from cs336_basics.llm_module.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.llm_module.position_wise_ffn import PositionWiseFeedForwardNetwork
from cs336_basics.llm_module.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,

        d_ff: int,

        num_heads: int,
        d_k: int = -1,
        d_v: int = -1,

        attn_pdrop: float | None=None,
        residual_pdrop: float | None=None,
    ):
        super().__init__()

        if d_k == -1: d_k = d_model // num_heads
        if d_v == -1: d_v = d_model // num_heads

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.attn_pdrop = attn_pdrop if attn_pdrop is not None else 0.0
        self.residual_pdrop = residual_pdrop if residual_pdrop is not None else 0.0



        self.norm_for_attention = RMSNorm(d_model, eps=1e-5)
        self.attention_block = MultiheadSelfAttention(d_model, num_heads, d_k, d_v, dropout=self.attn_pdrop)
        self.attention_residual_dropout = nn.Dropout(self.residual_pdrop)

        self.norm_for_ffn = RMSNorm(d_model, eps=1e-5)
        self.feed_forward_network = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.ffn_residual_dropout = nn.Dropout(self.residual_pdrop)


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.FloatTensor:
        # Multi-Head Self-Attention sub-layer
        # y = x + Dropout(MultiHeadSelfAttention(RMSNorm(x)))
        normed_x = self.norm_for_attention(x)
        attn_output = self.attention_block(normed_x, mask=mask)
        attn_output = self.attention_residual_dropout(attn_output)
        x = x + attn_output  # Residual connection

        # Position-wise Feed-Forward Network sub-layer
        normed_x = self.norm_for_ffn(x)
        ffn_output = self.feed_forward_network(normed_x)
        ffn_output = self.ffn_residual_dropout(ffn_output)
        x = x + ffn_output  # Residual connection

        return x


