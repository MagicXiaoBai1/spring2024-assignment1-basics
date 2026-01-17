from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
        state = self.state[p] # Get state associated with p.
        t = state.get("t", 0) # Get iteration number from the state, or initial value.
        grad = p.grad.data # Get the gradient of loss with respect to p.
        p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
        state["t"] = t + 1 # Increment iteration number.
        return loss



class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer (decoupled weight decay) as in:
    Loshchilov & Hutter, 2019. "Decoupled Weight Decay Regularization".

    Key property:
      - weight decay is applied directly to parameters (p *= (1 - lr*wd)),
        NOT added to the gradient, and NOT scaled by Adam's adaptive denom.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        # Basic argument validation (match PyTorch style)
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        closure: optional callable that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            # PyTorch convention: closure is evaluated with grad enabled
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("This AdamW implementation does not support sparse gradients.")

                state = self.state[p]

                # State initialization (one state per parameter tensor)
                if len(state) == 0:
                    state["step"] = 0
                    # First moment estimate (mean of gradients)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Second moment estimate (mean of squared gradients)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # ----------------------------
                # 1) Decoupled weight decay
                # ----------------------------
                # AdamW: apply weight decay directly to parameters, independent of gradient statistics.
                # p <- p - lr * wd * p   (equivalently p *= (1 - lr*wd))
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # ----------------------------
                # 2) Adam moment updates
                # ----------------------------
                # m_t = beta1*m_{t-1} + (1-beta1)*g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # v_t = beta2*v_{t-1} + (1-beta2)*(g_t^2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # ----------------------------
                # 3) Bias correction
                # ----------------------------
                # m_hat = m_t / (1 - beta1^t)
                # v_hat = v_t / (1 - beta2^t)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                # ----------------------------
                # 4) Parameter update
                # ----------------------------
                # denom = sqrt(v_hat) + eps
                # step_size = lr
                # p <- p - lr * m_hat / (sqrt(v_hat) + eps)
                #
                # To avoid explicitly forming m_hat and v_hat, we fold bias corrections into:
                # step_size = lr / bias_correction1
                # denom = sqrt(v_t)/sqrt(bias_correction2) + eps
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss