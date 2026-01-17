import torch
import torch.nn as nn

from cs336_basics.llm_module.position_embeddings.AbsolutePositionEmbeddings import AbsolutePositionEmbeddings
from cs336_basics.llm_module.rmsnorm import RMSNorm
from cs336_basics.llm_module.transformer_block import TransformerBlock


class TransformerLanguageModelOne(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,

            num_heads: int,
            d_ff: int,
            attn_pdrop: float,
            residual_pdrop: float
            ):
        super().__init__()

        # Token + position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = AbsolutePositionEmbeddings(d_model, context_length)
        self.dropout = nn.Dropout(p=residual_pdrop)

        self.layers = nn.Sequential()
        for i in range(num_layers):
            layer = TransformerBlock(d_model, d_ff, num_heads, attn_pdrop=attn_pdrop, residual_pdrop=residual_pdrop)
            self.layers.add_module(f"transformer_encoder_layer_{i}", layer)

        # Final normalization + LM head
        self.ln_final = RMSNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight  # Weight tying

        # Causal mask: True means "masked out".
        mask = torch.triu(torch.ones((1, 1, context_length, context_length), dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        # Expect token ids: (batch, seq_len)
        if x.dtype != torch.long:
            x = x.long()

        x = self.token_embeddings(x)  # (B, T, d_model)
        x = self.position_embeddings(x)
        x = self.dropout(x)

        # Truncate mask if the input is shorter than context_length
        seq_len = x.size(1)
        mask = self.mask[:, :, :seq_len, :seq_len]

        for sub_module in self.layers:
            x = sub_module(x, mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
