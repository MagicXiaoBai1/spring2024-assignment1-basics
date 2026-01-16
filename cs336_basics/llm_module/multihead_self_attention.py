import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.llm_module.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.llm_module.softmax import my_softmax


class MultiheadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,

                 n_head: int,
                 d_k: int = -1,
                 d_v: int = -1,

                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head

        if d_k == -1: d_k = d_model // n_head
        if d_v == -1: d_v = d_model // n_head

        self.d_k = d_k
        self.d_v = d_v


        # 3 次投影：Q、K、V
        self.Wq = nn.Linear(d_model, d_k*n_head, bias=False)
        self.Wk = nn.Linear(d_model, d_k*n_head, bias=False)
        self.Wv = nn.Linear(d_model, d_v*n_head, bias=False)        # 1 次输出投影
        self.Wo = nn.Linear(d_v*n_head, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (B, L, D)
        mask: (B, 1, 1, L)  True 表示要屏蔽（可选）
        """
        B, L, D = x.shape
        H, Dh, Dv = self.n_head, self.d_k, self.d_v

        # (B, L, D) -> (B, L, D)
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # reshape: (B, L, D) -> (B, H, L, Dh)
        q = q.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        k = k.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        v = v.view(B, L, H, Dv).transpose(1, 2)  # (B, H, L, Dh)

        # 注意力分数: (B, H, L, Dh) @ (B, H, Dh, L) -> (B, H, L, L)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn = my_softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # (B, H, L, L) @ (B, H, L, Dv) -> (B, H, L, Dv)
        out = torch.matmul(attn, v)

        # 合并 heads: (B, H, L, Dv) -> (B, L, H * Dv)
        out = out.transpose(1, 2).contiguous().view(B, L, H * Dv)

        # 输出投影: (B, L, H * Dv) -> (B, L, D)
        out = self.Wo(out)
        return out




def multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
    d_k: int = -1,
    d_v: int = -1,
) -> torch.FloatTensor:

    if d_k == -1: d_k = d_model // num_heads
    if d_v == -1: d_v = d_model // num_heads

    mha = MultiheadSelfAttention(d_model, num_heads, d_k, d_v, attn_pdrop)
    mha.Wo.weight.data = weights['output_proj.weight']
    for i in range(num_heads):
        mha.Wq.weight.data[i*d_k:(i+1)*d_k, :] = weights[f'q_heads.{i}.weight']
        mha.Wk.weight.data[i*d_k:(i+1)*d_k, :] = weights[f'k_heads.{i}.weight']
        mha.Wv.weight.data[i*d_v:(i+1)*d_v, :] = weights[f'v_heads.{i}.weight']

    mask = torch.triu(torch.ones(in_features.size(1), in_features.size(1), dtype=torch.bool), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(1)  # (1,
    return mha.forward(in_features, mask)


