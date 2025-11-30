import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class PowerReLU(nn.Module):
    """
    Novel activation function: ReLU(x^power) or x^power
    Applies power transformation followed by optional ReLU activation.

    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self, power: int, use_relu: bool = True):
        super().__init__()
        self.power = power
        self.use_relu = use_relu
        self.relu = nn.ReLU()

        # Statistics tracking
        self.pre_power_stats = {}
        self.post_power_stats = {}
        self.post_relu_stats = {}
        self.gradient_stats = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store pre-power input for gradient analysis
        if self.training and x.requires_grad:
            x.register_hook(lambda grad: self._log_pre_power_gradient(grad))
            self._log_forward_stats('pre_power', x)

        # Apply power transformation
        x_powered = torch.pow(x, self.power)

        if self.training and x_powered.requires_grad:
            x_powered.register_hook(lambda grad: self._log_post_power_gradient(grad))
            self._log_forward_stats('post_power', x_powered)

        # Apply ReLU if enabled
        if self.use_relu:
            output = self.relu(x_powered)
        else:
            output = x_powered

        if self.training:
            self._log_forward_stats('post_relu', output)

        return output

    def _log_forward_stats(self, stage: str, tensor: torch.Tensor):
        """Log forward pass statistics"""
        with torch.no_grad():
            stats = {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'zeros_pct': (tensor == 0).float().mean().item() * 100
            }

            if stage == 'pre_power':
                self.pre_power_stats = stats
            elif stage == 'post_power':
                self.post_power_stats = stats
            elif stage == 'post_relu':
                self.post_relu_stats = stats

    def _log_pre_power_gradient(self, grad: torch.Tensor):
        """Log gradient before power transformation"""
        with torch.no_grad():
            self.gradient_stats['pre_power'] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'norm': grad.norm().item(),
                'max_abs': grad.abs().max().item()
            }
        return grad

    def _log_post_power_gradient(self, grad: torch.Tensor):
        """Log gradient after power transformation"""
        with torch.no_grad():
            self.gradient_stats['post_power'] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'norm': grad.norm().item(),
                'max_abs': grad.abs().max().item()
            }
        return grad


class PositionwiseFeedForward(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1, layer_index: int = 0, use_relu: bool = True, use_post_power_norm: bool = False):
        super().__init__()

        # Store layer index for logging
        self.layer_index = layer_index
        self.use_post_power_norm = use_post_power_norm

        # Calculate power based on layer index (layer 0 -> power 1, layer 11 -> power 12)
        power = layer_index + 1

        # Calculate scaled initialization std: 0.02 / (power^(1/power))
        init_std = 0.02 / (power ** (1.0 / power))

        # Create layers
        self.linear1 = nn.Linear(dims, dims * rate)

        self.activation = PowerReLU(power, use_relu=use_relu)

        # Optional post-power normalization (after power transformation to prevent explosion)
        if use_post_power_norm:
            from gpt2.utils.fusing import LayerNorm
            self.post_power_norm = LayerNorm(dims * rate)

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dims * rate, dims)

        # Statistics tracking for post-norm activations
        self.post_norm_stats = {}

        # Apply smart weight initialization to both linear layers
        nn.init.normal_(self.linear1.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.linear2.bias)

    def get_gradient_stats(self):
        """Get comprehensive gradient and activation statistics for this layer"""
        return {
            'layer': self.layer_index,
            'power': self.activation.power,
            'forward': {
                'pre_power': self.activation.pre_power_stats,
                'post_power': self.activation.post_power_stats,
                'post_relu': self.activation.post_relu_stats,
                'post_norm': self.post_norm_stats if self.use_post_power_norm else {},
            },
            'gradients': self.activation.gradient_stats,
            'weight_grads': {
                'linear1': self.linear1.weight.grad.norm().item() if self.linear1.weight.grad is not None else 0,
                'linear2': self.linear2.weight.grad.norm().item() if self.linear2.weight.grad is not None else 0,
            }
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)

        # Optional normalization after power activation (prevents explosion)
        if self.use_post_power_norm:
            x = self.post_power_norm(x)

            # Log post-normalization statistics
            if self.training:
                with torch.no_grad():
                    self.post_norm_stats = {
                        'mean': x.mean().item(),
                        'std': x.std().item(),
                        'min': x.min().item(),
                        'max': x.max().item(),
                    }

        x = self.dropout(x)
        x = self.linear2(x)
        return x
