import torch
import torch.nn as nn


class AbsolutePositionEmbeddings(nn.Module):
    def __init__(self, d_model: int, context_length: int = 5000):
        """Absolute (learned) position embeddings.

        Stores a (context_length, d_model) table in `self.pe`.
        """
        super().__init__()
        self.d_model = d_model
        # Keep as a Parameter so the adapter/tests can assign into `.pe` directly.
        self.pe = nn.Parameter(torch.ones(context_length, d_model))

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        """Add positional embeddings to `x`.

        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # (1, seq_len, d_model) so it broadcasts over batch dimension.
        pos = self.pe[:seq_len, :].unsqueeze(0)
        return x + pos
