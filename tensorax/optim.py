"""
Optimization algorithms.
Pure C++/CUDA backend - no NumPy dependency.
"""

from typing import Iterator
from .tensor import Tensor


class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, params: Iterator[Tensor]):
        self.params = list(params)
    
    def step(self):
        """Perform a single optimization step."""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for param in self.params:
            param.grad = None


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, params: Iterator[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = {id(param): 0 for param in self.params}
    
    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            
            if self.momentum > 0:
                v = self.velocity[id(param)]
                if isinstance(v, (int, float)):
                    v = Tensor.zeros(param.shape, dtype=param.dtype, device=param.device)
                    self.velocity[id(param)] = v
                
                # v = momentum * v - lr * grad
                v_scaled = v * self.momentum
                grad_scaled = param.grad * self.lr
                v_new = v_scaled - grad_scaled
                self.velocity[id(param)] = v_new
                
                # param = param + v (in-place update)
                updated = param + v_new
                param._c_tensor = updated._c_tensor
                param._shape = updated._shape
                param._size = updated._size
            else:
                # param = param - lr * grad (in-place update)
                grad_scaled = param.grad * self.lr
                updated = param - grad_scaled
                param._c_tensor = updated._c_tensor
                param._shape = updated._shape
                param._size = updated._size


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(
        self, 
        params: Iterator[Tensor], 
        lr: float = 0.001, 
        betas: tuple = (0.9, 0.999), 
        eps: float = 1e-8
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = {id(param): 0 for param in self.params}
        self.v = {id(param): 0 for param in self.params}
    
    def step(self):
        self.t += 1
        
        for param in self.params:
            if param.grad is None:
                continue
            
            param_id = id(param)
            
            # Initialize moments if needed
            if isinstance(self.m[param_id], (int, float)):
                self.m[param_id] = Tensor.zeros(param.shape, dtype=param.dtype, device=param.device)
                self.v[param_id] = Tensor.zeros(param.shape, dtype=param.dtype, device=param.device)
            
            grad = param.grad
            
            # Update biased first moment estimate: m = beta1 * m + (1-beta1) * grad
            m_old = self.m[param_id]
            self.m[param_id] = (m_old * self.beta1 + grad * (1 - self.beta1))
            
            # Update biased second raw moment estimate: v = beta2 * v + (1-beta2) * grad^2
            v_old = self.v[param_id]
            grad_sq = grad * grad
            self.v[param_id] = (v_old * self.beta2 + grad_sq * (1 - self.beta2))
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** self.t
            bias_correction2 = 1 - self.beta2 ** self.t
            
            # m_hat = m / bias_correction1
            m_hat = self.m[param_id] * (1.0 / bias_correction1)
            
            # v_hat = v / bias_correction2
            v_hat = self.v[param_id] * (1.0 / bias_correction2)
            
            # param = param - lr * m_hat / (sqrt(v_hat) + eps)
            # Use C++ sqrt for performance
            v_hat_sqrt = v_hat.sqrt()
            
            # Add epsilon for numerical stability
            denominator = v_hat_sqrt + self.eps
            
            # Compute update
            update = m_hat / denominator
            
            # Update parameter: param = param - lr * update
            updated = param - update * self.lr
            param._c_tensor = updated._c_tensor
            param._shape = updated._shape
            param._size = updated._size
            # For now, placeholder for structure
            # param._c_tensor = (param - m_hat * self.lr)._c_tensor  # Simplified
