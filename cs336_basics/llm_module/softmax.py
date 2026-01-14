import torch


def my_softmax(in_features: torch.FloatTensor, dim: int):
    """
    Compute the softmax of the input tensor along the specified dimension.

    Args:
        in_features (torch.FloatTensor): Input tensor.
        dim (int): Dimension along which to compute the softmax.

    Returns:
        torch.FloatTensor: Softmax of the input tensor along the specified dimension.
    """
    # Subtract the max for numerical stability
    max_vals, _ = torch.max(in_features, dim=dim, keepdim=True)
    exps = torch.exp(in_features - max_vals)
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    softmax = exps / sum_exps
    return softmax