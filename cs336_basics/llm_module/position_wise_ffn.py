import torch

from cs336_basics.llm_module.activation_function import gaussian_error_linear_units


class PositionWiseFeedForwardNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_inner = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_outer = torch.nn.Linear(hidden_dim, input_dim, bias=False)
        self.activation = gaussian_error_linear_units

    def forward(self, x):
        hidden = self.hidden_inner(x)
        activated = self.activation(hidden)
        output = self.hidden_outer(activated)
        return output