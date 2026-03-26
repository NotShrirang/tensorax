"""
Neural network layers.
Pure C++/CUDA backend - no NumPy dependency.
"""

from typing import Optional
from .module import Module
from ..tensor import Tensor
from .. import functional as F
import random
import math


class Linear(Module):
    """Fully connected linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with Xavier/Glorot initialization
        # Using pure Python random for now (will be replaced by C++ random)
        std = math.sqrt(2.0 / (in_features + out_features))
        weight_data = []
        for _ in range(out_features * in_features):
            weight_data.append(random.gauss(0, std))
        
        self._parameters['weight'] = Tensor(
            weight_data,
            shape=(out_features, in_features),
            requires_grad=True
        )
        
        if bias:
            self._parameters['bias'] = Tensor.zeros(
                (out_features,),
                requires_grad=True
            )
        else:
            self._parameters['bias'] = None
    
    @property
    def weight(self) -> Tensor:
        """Get weight parameter."""
        return self._parameters['weight']
    
    @property
    def bias(self) -> Optional[Tensor]:
        """Get bias parameter."""
        return self._parameters['bias']
    
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self._parameters['weight'], self._parameters['bias'])


class ReLU(Module):
    """ReLU activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class Sigmoid(Module):
    """Sigmoid activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Tanh(Module):
    """Tanh activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class Softmax(Module):
    """Softmax activation layer."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, dim=self.dim)


class Dropout(Module):
    """Dropout layer."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        # Create dropout mask using Python random for now
        # TODO: Implement in C++ for better performance
        mask_data = []
        scale = 1.0 / (1.0 - self.p)
        for _ in range(x.size):
            mask_data.append(scale if random.random() > self.p else 0.0)
        
        mask = Tensor(mask_data, shape=x.shape, device=x.device)
        return x * mask


class Sequential(Module):
    """Sequential container for modules."""
    
    def __init__(self, *layers):
        super().__init__()
        # Handle both Sequential(layer1, layer2) and Sequential([layer1, layer2])
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]
        if not isinstance(layers, (list, tuple)):
            raise ValueError("Layers must be provided as a list or tuple")
        
        for idx, layer in enumerate(layers):
            if not isinstance(layer, Module):
                raise AttributeError(f"Expected Module instance, got {type(layer)}")
            self._modules[str(idx)] = layer
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
    
    def __getitem__(self, index: int) -> Module:
        """Get layer by index."""
        if str(index) not in self._modules:
            raise IndexError("Index out of range")
        return self._modules[str(index)]


class RMSNorm(Module):
    """Root Mean Square Normalization."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self._parameters['weight'] = Tensor.ones((normalized_shape,), requires_grad=True)
        
    def forward(self, x: Tensor) -> Tensor:
        # Calculate RMS: sqrt(mean(x^2))
        variance = (x * x).mean(dim=-1, keepdim=True)
        x_normed = x * (variance + self.eps).sqrt() ** -1.0
        return x_normed * self._parameters['weight']

class LayerNorm(Module):
    """Layer Normalization."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self._parameters['weight'] = Tensor.ones((normalized_shape,), requires_grad=True)
        self._parameters['bias'] = Tensor.zeros((normalized_shape,), requires_grad=True)
        
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
        x_normed = (x - mean) * (variance + self.eps).sqrt() ** -1.0
        return x_normed * self._parameters['weight'] + self._parameters['bias']

class BatchNorm(Module):
    """Batch Normalization (1D)."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self._parameters['weight'] = Tensor.ones((num_features,), requires_grad=True)
        self._parameters['bias'] = Tensor.zeros((num_features,), requires_grad=True)
        # Using tensors for running stats without requires_grad
        self.running_mean = Tensor.zeros((num_features,), requires_grad=False)
        self.running_var = Tensor.ones((num_features,), requires_grad=False)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            variance = ((x - mean) * (x - mean)).mean(dim=0, keepdim=True)

            self.running_mean = self.running_mean * (1 - self.momentum) + mean.sum(dim=0) * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + variance.sum(dim=0) * self.momentum
        else:
            mean = self.running_mean
            variance = self.running_var
            
        x_normed = (x - mean) * (variance + self.eps).sqrt() ** -1.0
        return x_normed * self._parameters['weight'] + self._parameters['bias']
