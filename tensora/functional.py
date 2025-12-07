"""
Functional API for tensor operations.
Pure C++/CUDA backend - no NumPy dependency.
"""

from .tensor import Tensor
try:
    from . import _C
except ImportError:
    _C = None


def relu(x: Tensor) -> Tensor:
    """ReLU activation function: max(0, x)."""
    result = Tensor.__new__(Tensor)
    result._shape = x._shape
    result._size = x._size
    result.dtype = x.dtype
    result.device = x.device
    result.grad = None
    
    if _C:
        result._c_tensor = _C.relu(x._c_tensor)
    
    if x.requires_grad:
        result.requires_grad = True
        result._grad_fn = ('relu', x)
    else:
        result.requires_grad = False
        result._grad_fn = None
    
    return result


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation function: 1 / (1 + exp(-x))."""
    result = Tensor.__new__(Tensor)
    result._shape = x._shape
    result._size = x._size
    result.dtype = x.dtype
    result.device = x.device
    result.grad = None
    
    if _C:
        result._c_tensor = _C.sigmoid(x._c_tensor)
    
    if x.requires_grad:
        result.requires_grad = True
        result._grad_fn = ('sigmoid', x, result)
    else:
        result.requires_grad = False
        result._grad_fn = None
    
    return result


def tanh(x: Tensor) -> Tensor:
    """Tanh activation function."""
    result = Tensor.__new__(Tensor)
    result._shape = x._shape
    result._size = x._size
    result.dtype = x.dtype
    result.device = x.device
    result.grad = None
    
    if _C:
        result._c_tensor = _C.tanh(x._c_tensor)
    
    if x.requires_grad:
        result.requires_grad = True
        result._grad_fn = ('tanh', x, result)
    else:
        result.requires_grad = False
        result._grad_fn = None
    
    return result


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax activation function."""
    if dim < 0:
        dim += len(x._shape)

    result = Tensor.__new__(Tensor)
    result._shape = x._shape
    result._size = x._size
    result.dtype = x.dtype
    result.device = x.device
    result.grad = None
    
    if _C:
        result._c_tensor = _C.softmax(x._c_tensor, dim)
    
    if x.requires_grad:
        result.requires_grad = True
        result._grad_fn = ('softmax', x, result, dim)
    else:
        result.requires_grad = False
        result._grad_fn = None
    
    return result


def linear(x: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """Linear transformation."""
    output = x @ weight.T
    if bias is not None:
        output = output + bias
    return output


def conv2d(x: Tensor, weight: Tensor, bias: Tensor = None, stride: int = 1, padding: int = 0) -> Tensor:
    """2D convolution (placeholder for CUDA implementation)."""
    raise NotImplementedError("Conv2D will be implemented with CUDA kernels")


def max_pool2d(x: Tensor, kernel_size: int, stride: int = None) -> Tensor:
    """2D max pooling (placeholder for CUDA implementation)."""
    raise NotImplementedError("MaxPool2D will be implemented with CUDA kernels")


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss."""
    result = Tensor.__new__(Tensor)
    result._shape = ()  # Scalar
    result._size = 1
    result.dtype = pred.dtype
    result.device = pred.device
    result.grad = None
    
    if _C:
        result._c_tensor = _C.mse_loss(pred._c_tensor, target._c_tensor)
    
    if pred.requires_grad:
        result.requires_grad = True
        result._grad_fn = ('mse_loss', pred, target)
    else:
        result.requires_grad = False
        result._grad_fn = None
    
    return result


def cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Cross entropy loss."""
    result = Tensor.__new__(Tensor)
    result._shape = ()  # Scalar
    result._size = 1
    result.dtype = pred.dtype
    result.device = pred.device
    result.grad = None
    
    if _C:
        result._c_tensor = _C.cross_entropy_loss(pred._c_tensor, target._c_tensor)
    
    if pred.requires_grad:
        result.requires_grad = True
        result._grad_fn = ('cross_entropy', pred, target)
    else:
        result.requires_grad = False
        result._grad_fn = None
    
    return result
