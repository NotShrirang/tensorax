"""
Functional API for tensor operations.
Pure C++/CUDA backend - no NumPy dependency.
"""

from typing import Optional
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

    if pred._shape != target._shape:
        raise RuntimeError(f"MSE Loss: Shape mismatch between pred {pred._shape} and target {target._shape}. Tensora does not support broadcasting in loss functions.")

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
    """Cross entropy loss (expects softmax probabilities and one-hot targets)."""
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


def cross_entropy_from_logits(logits: Tensor, targets: Tensor, reduce_mean: bool = True) -> Tensor:
    """
    Cross entropy loss from raw logits (more numerically stable).
    
    Args:
        logits: Raw predictions before softmax. Shape: (batch_size, num_classes) or (num_classes,)
        targets: Class indices. Shape: (batch_size,) or scalar
        reduce_mean: If True, returns mean loss. If False, returns per-sample losses.
    
    Returns:
        Loss tensor (scalar if reduce_mean=True, else (batch_size,))
    
    Example:
        >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.0]])  # batch_size=2, num_classes=3
        >>> targets = Tensor([0, 1])  # class indices
        >>> loss = F.cross_entropy_from_logits(logits, targets)
    """
    result = Tensor.__new__(Tensor)
    
    if _C:
        result._c_tensor = _C.cross_entropy_from_logits(logits._c_tensor, targets._c_tensor, reduce_mean)
    
    if reduce_mean:
        result._shape = ()  # Scalar
        result._size = 1
    else:
        # Per-sample losses
        if len(logits._shape) == 1:
            result._shape = ()
            result._size = 1
        else:
            result._shape = (logits._shape[0],)
            result._size = logits._shape[0]
    
    result.dtype = logits.dtype
    result.device = logits.device
    result.grad = None
    
    if logits.requires_grad:
        result.requires_grad = True
        result._grad_fn = ('cross_entropy_from_logits', logits, targets, reduce_mean)
    else:
        result.requires_grad = False
        result._grad_fn = None
    
    return result



def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Scaled Dot-Product Attention.
    
    Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        query: Query tensor [batch_size, num_heads, seq_len_q, d_k]
        key: Key tensor [batch_size, num_heads, seq_len_k, d_k]
        value: Value tensor [batch_size, num_heads, seq_len_v, d_v]
        mask: Optional attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
              Use -inf to mask out positions (they'll become 0 after softmax)
    
    Returns:
        output: Attention output [batch_size, num_heads, seq_len_q, d_v]
    
    Example:
        >>> batch, heads, seq_len, d_k, d_v = 2, 8, 10, 64, 64
        >>> Q = Tensor.randn(batch, heads, seq_len, d_k)
        >>> K = Tensor.randn(batch, heads, seq_len, d_k)
        >>> V = Tensor.randn(batch, heads, seq_len, d_v)
        >>> out = F.scaled_dot_product_attention(Q, K, V)
        >>> out.shape  # [2, 8, 10, 64]
    """
    # TODO: Implement Python wrapper for scaled_dot_product_attention
    # 
    # Steps:
    # 1. Validate inputs (shape, device compatibility)
    # 2. Create result tensor with proper shape
    # 3. Call C++ backend through _C module
    # 4. Set up gradient function if requires_grad=True
    # 5. Return result
    
    # Validate shapes
    if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
        raise ValueError("Query, Key, and Value must be 4D tensors [batch, heads, seq_len, d]")
    
    if query.device != key.device or query.device != value.device:
        raise ValueError("All tensors must be on the same device")
    
    if mask is not None and mask.device != query.device:
        raise ValueError("Mask must be on the same device as Q, K, V")
    
    # Extract dimensions
    batch_size = query.shape[0]
    num_heads = query.shape[1]
    seq_len_q = query.shape[2]
    d_k = query.shape[3]
    d_v = value.shape[3]
    
    # Create result tensor
    result = Tensor.__new__(Tensor)
    result._shape = (batch_size, num_heads, seq_len_q, d_v)
    result._size = batch_size * num_heads * seq_len_q * d_v
    result.dtype = query.dtype
    result.device = query.device
    result.grad = None
    
    # Call C++ backend
    if _C:
        mask_c_tensor = mask._c_tensor if mask is not None else None
        result._c_tensor = _C.scaled_dot_product_attention(
            query._c_tensor,
            key._c_tensor,
            value._c_tensor,
            mask_c_tensor
        )
    else:
        raise RuntimeError("C++ backend not available. Please build the package first.")
    
    # Set up autograd
    if query.requires_grad or key.requires_grad or value.requires_grad:
        result.requires_grad = True
        result._grad_fn = ('scaled_dot_product_attention', query, key, value, mask)
    else:
        result.requires_grad = False
        result._grad_fn = None

    return result


def _sdpa_variant(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], backend_fn):
    if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
        raise ValueError("Query, Key, and Value must be 4D tensors [batch, heads, seq_len, d]")
    if query.device != key.device or query.device != value.device:
        raise ValueError("All tensors must be on the same device")
    if mask is not None and mask.device != query.device:
        raise ValueError("Mask must be on the same device as Q, K, V")

    batch_size = query.shape[0]
    num_heads = query.shape[1]
    seq_len_q = query.shape[2]
    d_v = value.shape[3]

    result = Tensor.__new__(Tensor)
    result._shape = (batch_size, num_heads, seq_len_q, d_v)
    result._size = batch_size * num_heads * seq_len_q * d_v
    result.dtype = query.dtype
    result.device = query.device
    result.grad = None

    if _C:
        mask_c = mask._c_tensor if mask is not None else None
        result._c_tensor = backend_fn(query._c_tensor, key._c_tensor, value._c_tensor, mask_c)
    else:
        raise RuntimeError("C++ backend not available.")

    result.requires_grad = False
    result._grad_fn = None
    return result


def scaled_dot_product_attention_tiled(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    return _sdpa_variant(query, key, value, mask, _C.scaled_dot_product_attention_tiled)


def scaled_dot_product_attention_flash(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    return _sdpa_variant(query, key, value, mask, _C.scaled_dot_product_attention_flash)
