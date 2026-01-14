import torch

def me_erf(in_features: torch.FloatTensor):
    # Approximation of the error function
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = torch.sign(in_features)
    x = torch.abs(in_features)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * torch.exp(-x * x)

    return sign * y

def gaussian_error_linear_units(in_features: torch.FloatTensor):
    return 0.5 * in_features * (1.0 + me_erf(in_features / torch.sqrt(torch.tensor(2.0))))


def gaussian_error_linear_units2(in_features: torch.FloatTensor):
    return 0.5 * in_features * (1.0 + torch.tanh(torch.sqrt(2 / torch.pi) * (in_features + 0.044715 * torch.pow(in_features, 3))))
