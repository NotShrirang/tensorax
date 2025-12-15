"""
Neural network module base classes.
"""

from typing import Iterator, Tuple
from ..tensor import Tensor


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Make module callable."""
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> Iterator[Tensor]:
        """Return iterator over module parameters."""
        for param in self._parameters.values():
            if param is not None:
                yield param
            else:
                continue
        for module in self._modules.values():
            if module is not None:
                yield from module.parameters()
            else:
                continue
    
    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Return iterator over module parameters with names."""
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param
    
    def train(self, mode: bool = True) -> 'Module':
        """Set module to training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set module to evaluation mode."""
        return self.train(False)
    
    def  cuda(self) -> 'Module':
        """Move all parameters to CUDA."""
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = param.cuda()
        for module in self._modules.values():
            module.cuda()
        return self
    
    def cpu(self) -> 'Module':
        """Move all parameters to CPU."""
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = param.cpu()
        for module in self._modules.values():
            module.cpu()
        return self
    
    def to(self, device: str) -> 'Module':
        """Move module to specified device."""
        if device == 'cuda':
            return self.cuda()
        elif device == 'cpu':
            return self.cpu()
        else:
            raise ValueError(f"Unknown device: {device}")
    
    def zero_grad(self) -> None:
        """Zero out gradients of all parameters."""
        for param in self.parameters():
            param.grad = None

    def __repr__(self):
        return f"<{self.__class__.__name__} training={self.training} parameters={list(self.parameters())}>"
    
    def __str__(self):
        return self.__repr__()
