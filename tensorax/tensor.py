"""
Core Tensor class for Tensorax library.
Pure C++/CUDA backend - no NumPy dependency.
"""

from typing import Iterable, Union, Tuple, Optional, List
import warnings

try:
    from . import _C
except ImportError as e:
    warnings.warn(f"Failed to import _C module: {e}. Tensor operations will not work until the package is built.")
    _C = None  # Will be available after building

import tensorax as ts
from tensorax.utils.type_checks import is_numpy_array
from tensorax.utils.shape_utils import _compute_size, _has_valid_shape

class Tensor:
    """
    Multi-dimensional array with automatic differentiation support.
    
    Similar to PyTorch tensors but with custom CUDA kernels for optimal performance.
    """
    
    def __init__(
        self, 
        data: Union[list, 'Tensor'],
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[str] = None,
        device: str = 'cpu',
        requires_grad: bool = False
    ) -> None:
        """
        Initialize a Tensor.
        
        Args:
            data: Input data (list, flat buffer, or another Tensor)
            shape: Shape of the tensor (inferred from list if not provided)
            dtype: Data type ('float32', 'float64', 'int32', 'int64')
            device: Device to place tensor on ('cpu' or 'cuda')
            requires_grad: Whether to track gradients
        """
        # Validate device before doing anything else
        if device not in (ts.cpu, ts.cuda):
            raise ValueError(f"Unknown device: {device}")
        
        if dtype is not None and dtype not in ts.valid_tensor_dtypes:
            raise ValueError(f"Invalid dtype: {dtype}. Must be one of {ts.valid_tensor_dtypes}")
        
        if shape is None and not _has_valid_shape(data):
            raise ValueError("Cannot infer shape from data: inconsistent nested list lengths. Either provide a valid shape or ensure data is a well-formed nested list.")

        self.dtype = dtype or 'float32'
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        
        if isinstance(data, Tensor):
            # Copy from another tensor
            self._shape = data._shape
            self._size = data._size
            self.dtype = data.dtype
            self.device = data.device
            self.requires_grad = data.requires_grad
            self._c_tensor = _C.copy_tensor(data._c_tensor) if _C else None
        else:
            # Create from list/data
            flat_data, inferred_shape = self._flatten_data(data)
            if shape is not None:
                size = _compute_size(shape)
                if size != len(flat_data):
                    raise ValueError(f"Data size {len(flat_data)} does not match provided shape {shape} (size {size})")
            self._shape = shape or inferred_shape
            self._size = self._compute_size(self._shape)
            
            # Create C++ tensor
            if _C:
                if device == ts.cpu:
                    self._c_tensor = _C.create_tensor_cpu(flat_data, list(self._shape), self.dtype)
                elif device == ts.cuda:
                    self._c_tensor = _C.create_tensor_cuda(flat_data, list(self._shape), self.dtype)
                else:
                    raise ValueError(f"Unknown device: {device}")
            else:
                self._c_tensor = None
                self._data = flat_data  # Fallback for testing before build

    @staticmethod
    def cuda_is_available() -> bool:
        """Check if CUDA is available."""
        if _C:
            return _C.cuda_is_available()
        return False

    @staticmethod
    def _flatten_data(data):
        """Flatten nested lists and infer shape."""
        def get_shape(d):
            if not isinstance(d, list):
                return []
            if len(d) == 0:
                return [0]
            return [len(d)] + get_shape(d[0])
        
        def flatten(d):
            if not isinstance(d, list):
                return [float(d)]
            result = []
            for item in d:
                result.extend(flatten(item))
            return result
        
        if is_numpy_array(data):
            shape = data.shape
            flat = data.flatten().tolist()
            return flat, shape

        shape = tuple(get_shape(data))
        flat = flatten(data)
        return flat, shape
    
    @staticmethod
    def _compute_size(shape):
        """Compute total number of elements."""
        return _compute_size(shape)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self._shape
    
    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self._shape)
    
    @property
    def size(self) -> int:
        """Return total number of elements."""
        return self._size
    
    def tolist(self) -> list:
        """Convert to nested Python list."""
        if _C is None:
            return self._data
        
        flat_data = _C.tensor_to_list(self._c_tensor)
        return self._unflatten_data(flat_data, self._shape)
    
    @staticmethod
    def _unflatten_data(flat_data, shape):
        """Reshape flat list to nested structure."""
        if len(shape) == 0:
            return flat_data[0]
        if len(shape) == 1:
            return flat_data
        
        size = 1
        for dim in shape[1:]:
            size *= dim
        
        result = []
        for i in range(shape[0]):
            start = i * size
            end = start + size
            result.append(Tensor._unflatten_data(flat_data[start:end], shape[1:]))
        return result
    
    def cpu(self) -> 'Tensor':
        """Move tensor to CPU."""
        if self.device == 'cpu':
            return self
        
        if _C:
            new_tensor = Tensor.__new__(Tensor)
            new_tensor._shape = self._shape
            new_tensor._size = self._size
            new_tensor.dtype = self.dtype
            new_tensor.device = 'cpu'
            new_tensor.requires_grad = self.requires_grad
            new_tensor.grad = None
            new_tensor._grad_fn = self._grad_fn
            new_tensor._c_tensor = _C.tensor_cuda_to_cpu(self._c_tensor)
            return new_tensor
        return self
    
    def cuda(self) -> 'Tensor':
        """Move tensor to CUDA."""
        if self.device == 'cuda':
            return self
        
        if not _C or not _C.cuda_is_available():
            raise RuntimeError("CUDA is not available")
        
        new_tensor = Tensor.__new__(Tensor)
        new_tensor._shape = self._shape
        new_tensor._size = self._size
        new_tensor.dtype = self.dtype
        new_tensor.device = 'cuda'
        new_tensor.requires_grad = self.requires_grad
        new_tensor.grad = None
        new_tensor._grad_fn = self._grad_fn
        new_tensor._c_tensor = _C.tensor_cpu_to_cuda(self._c_tensor)
        return new_tensor
    
    def to(self, device: str) -> 'Tensor':
        """Move tensor to specified device."""
        if device == 'cpu':
            return self.cpu()
        elif device == 'cuda':
            return self.cuda()
        else:
            raise ValueError(f"Unknown device: {device}")
    
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise addition."""
        if isinstance(other, (int, float)):
            other = Tensor.full(self._shape, other, dtype=self.dtype, device=self.device)
        
        if self.device != other.device:
            raise RuntimeError(f"Tensors on different devices: {self.device} vs {other.device}")

        shape_a = self._shape
        shape_b = other._shape

        if _C:
            if shape_a == shape_b:
                result = Tensor.__new__(Tensor)
                result._shape = self._shape
                result._size = self._size
                result.dtype = self.dtype
                result.device = self.device
                result.grad = None
                result._c_tensor = _C.add(self._c_tensor, other._c_tensor)
            else:
                # Broadcasting case
                result_shape = []
                len_a = len(shape_a)
                len_b = len(shape_b)
                for i in range(max(len_a, len_b)):
                    dim_a = shape_a[-(i+1)] if i < len_a else 1
                    dim_b = shape_b[-(i+1)] if i < len_b else 1
                    if dim_a != dim_b and dim_a != 1 and dim_b != 1:
                        raise RuntimeError(f"Incompatible shapes for broadcasting: {shape_a} and {shape_b}")
                    result_shape.insert(0, max(dim_a, dim_b))
                
                result = Tensor.__new__(Tensor)
                result._shape = tuple(result_shape)
                result._size = Tensor._compute_size(result_shape)
                result.dtype = self.dtype
                result.device = self.device
                result.grad = None

                if shape_a < shape_b:
                    result._c_tensor = _C.broadcasting_add(self._c_tensor, other._c_tensor)
                else:
                    result._c_tensor = _C.broadcasting_add(other._c_tensor, self._c_tensor)
        
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('add', self, other)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Right-hand element-wise addition."""
        return self.__add__(other)
    
    def __rmul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Right-hand element-wise multiplication."""
        return self.__mul__(other)

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise multiplication."""
        if isinstance(other, (int, float)):
            other = Tensor.full(self._shape, other, dtype=self.dtype, device=self.device)
        
        if self.device != other.device:
            raise RuntimeError(f"Tensors on different devices: {self.device} vs {other.device}")
        
        result = Tensor.__new__(Tensor)
        result._shape = self._shape
        result._size = self._size
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.multiply(self._c_tensor, other._c_tensor)
        
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('mul', self, other)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise subtraction."""
        if isinstance(other, (int, float)):
            other = Tensor.full(self._shape, other, dtype=self.dtype, device=self.device)
        
        if self.device != other.device:
            raise RuntimeError(f"Tensors on different devices: {self.device} vs {other.device}")
        
        result = Tensor.__new__(Tensor)
        result._shape = self._shape
        result._size = self._size
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.subtract(self._c_tensor, other._c_tensor)
        
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('sub', self, other)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Element-wise division."""
        if isinstance(other, (int, float)):
            other = Tensor.full(self._shape, other, dtype=self.dtype, device=self.device)
        
        if self.device != other.device:
            raise RuntimeError(f"Tensors on different devices: {self.device} vs {other.device}")
        
        result = Tensor.__new__(Tensor)
        result._shape = self._shape
        result._size = self._size
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.divide(self._c_tensor, other._c_tensor)
        
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('div', self, other)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def matmul(self, other: 'Tensor', method: str = "default") -> 'Tensor':
        """Matrix multiplication.
        
        Args:
            other: Another tensor to multiply with
            method: Method for CUDA matmul ('default', 'shared_memory_coalesced', 'tiled')

        Returns:
            Resulting tensor from matrix multiplication
        """

        if self.device != other.device:
            raise RuntimeError(f"Tensors on different devices: {self.device} vs {other.device}")
        
        if len(self._shape) < 2 or len(other._shape) < 2:
            raise RuntimeError(f"Matrix multiplication requires 2D+ tensors, got {self._shape} and {other._shape}")
        
        if self._shape[-1] != other._shape[-2]:
            raise RuntimeError(f"Incompatible shapes for matmul: {self._shape} and {other._shape}. {self._shape[-1]} != {other._shape[-2]}")

        result_shape = self._shape[:-1] + (other._shape[-1],)
        
        result = Tensor.__new__(Tensor)
        result._shape = result_shape
        result._size = self._compute_size(result_shape)
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = self._matmul(self._c_tensor, other._c_tensor, method=method)
        
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('matmul', self, other)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def _matmul(self, a_c_tensor, b_c_tensor, method: str = "default") -> 'Tensor':
        """Internal method to perform matrix multiplication using C++ backend."""
        if _C:
            if self.device == 'cpu':
                if method != "default":
                    warnings.warn(f"Matrix multiplication method '{method}' not supported on CPU. Using default method.")
                return _C.matmul(a_c_tensor, b_c_tensor)
            elif self.device == 'cuda':
                if method == "default":
                    return _C.matmul(a_c_tensor, b_c_tensor)
                elif method == "shared_memory_coalesced":
                    return _C.matmul_with_shared_memory_coalescing(a_c_tensor, b_c_tensor, 1.0, 0.0)
                elif method == "tiled":
                    return _C.matmul_tiled(a_c_tensor, b_c_tensor)
                elif method == "shared_memory_cache_blocking":
                    return _C.matmul_with_shared_memory_cache_blocking(a_c_tensor, b_c_tensor, 1.0, 0.0)
                else:
                    raise ValueError(f"Unknown matmul method: {method}")
        raise RuntimeError("C++ extension not built. Matrix multiplication not available.")
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        if self.device != other.device:
            raise RuntimeError(f"Tensors on different devices: {self.device} vs {other.device}")
        
        if len(self._shape) < 2 or len(other._shape) < 2:
            raise RuntimeError(f"Matrix multiplication requires 2D+ tensors, got {self._shape} and {other._shape}")
        
        if self._shape[-1] != other._shape[-2]:
            raise RuntimeError(f"Incompatible shapes for matmul: {self._shape} and {other._shape}. {self._shape[-1]} != {other._shape[-2]}")

        result_shape = self._shape[:-1] + (other._shape[-1],)
        
        result = Tensor.__new__(Tensor)
        result._shape = result_shape
        result._size = self._compute_size(result_shape)
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = self._matmul(self._c_tensor, other._c_tensor)
        
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('matmul', self, other)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def __repr__(self) -> str:
        data_list = self.tolist() if _C else "<not built>"
        for _line in repr(data_list).splitlines():
            data_repr = _line
            break  # Just take the first line for brevity
        data_repr = data_repr.replace('\n', ' ') + (' ...' if len(repr(data_list).splitlines()) > 1 else '')
        
        return f"Tensor({data_repr}, device='{self.device}', dtype='{self.dtype}')"
    
    def __str__(self) -> str:
        return self.__repr__()

    def __pow__(self, power: float) -> 'Tensor':
        """Element-wise power."""
        result = Tensor.__new__(Tensor)
        result._shape = self._shape
        result._size = self._size
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.pow(self._c_tensor, float(power))
        
        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('pow', self, power)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype: str = 'float32', device: str = 'cpu', requires_grad: bool = False) -> 'Tensor':
        """Create a tensor filled with zeros."""
        size = Tensor._compute_size(shape)
        data = [0.0] * size
        return Tensor(data, shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], dtype: str = 'float32', device: str = 'cpu', requires_grad: bool = False) -> 'Tensor':
        """Create a tensor filled with ones."""
        size = Tensor._compute_size(shape)
        data = [1.0] * size
        return Tensor(data, shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
    
    @staticmethod
    def full(shape: Tuple[int, ...], value: float, dtype: str = 'float32', device: str = 'cpu', requires_grad: bool = False) -> 'Tensor':
        """Create a tensor filled with a specific value."""
        size = Tensor._compute_size(shape)
        data = [float(value)] * size
        return Tensor(data, shape=shape, dtype=dtype, device=device, requires_grad=requires_grad)
    
    @staticmethod
    def randn(shape: Tuple[int, ...], dtype: str = 'float32', device: str = 'cpu', requires_grad: bool = False) -> 'Tensor':
        """Create a tensor with random values from normal distribution (requires C++ random)."""
        if _C:
            tensor = Tensor.__new__(Tensor)
            tensor._shape = shape
            tensor._size = Tensor._compute_size(shape)
            tensor.dtype = dtype
            tensor.device = device
            tensor.requires_grad = requires_grad
            tensor.grad = None
            tensor._grad_fn = None
            tensor._c_tensor = _C.randn(list(shape), dtype, device)
            return tensor
        raise RuntimeError("C++ extension not built. Use simple initialization instead.")
    
    def zero_grad(self):
        """Zero out the gradients."""
        self.grad = None
    
    def backward(self, grad: Optional['Tensor'] = None):
        """Compute gradients via backpropagation."""
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = Tensor.ones(self._shape, device=self.device)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad
        
        if self._grad_fn is not None:
            op, *inputs = self._grad_fn
            
            if op == 'add':
                if inputs[0].requires_grad:
                    grad_input = grad
                    # Handle broadcasting: sum over dimensions that were broadcast
                    if inputs[0]._shape != grad._shape:
                        # Need to sum and reshape gradient to match input shape
                        for i in range(len(grad._shape)):
                            if i >= len(inputs[0]._shape) or inputs[0]._shape[i] == 1:
                                grad_input = grad_input.sum(dim=i)
                    inputs[0].backward(grad_input)
                if inputs[1].requires_grad:
                    grad_input = grad
                    # Handle broadcasting: sum over dimensions that were broadcast
                    if inputs[1]._shape != grad._shape:
                        # Need to sum gradient over dimensions that were broadcast
                        axes_to_sum = []
                        # Determine which axes to sum over
                        len_diff = len(grad._shape) - len(inputs[1]._shape)
                        for i in range(len(grad._shape)):
                            if i < len_diff:
                                # This dimension doesn't exist in input[1], sum it
                                axes_to_sum.append(i)
                            else:
                                # Check if this dimension was broadcast (size 1 in input)
                                input_idx = i - len_diff
                                if input_idx < len(inputs[1]._shape) and inputs[1]._shape[input_idx] == 1 and grad._shape[i] > 1:
                                    axes_to_sum.append(i)
                        
                        # Sum over all necessary axes
                        for axis in sorted(axes_to_sum, reverse=True):
                            grad_input = grad_input.sum(dim=axis)
                        
                        # Reshape if needed to match original shape
                        if grad_input._shape != inputs[1]._shape:
                            # For now, just ensure the gradient has the right shape
                            pass
                    inputs[1].backward(grad_input)
            elif op == 'sub':
                if inputs[0].requires_grad:
                    inputs[0].backward(grad)
                if inputs[1].requires_grad:
                    inputs[1].backward(grad * Tensor.full(grad.shape, -1.0, device=grad.device))
            elif op == 'mul':
                if inputs[0].requires_grad:
                    inputs[0].backward(grad * inputs[1])
                if inputs[1].requires_grad:
                    inputs[1].backward(grad * inputs[0])
            elif op == 'div':
                if inputs[0].requires_grad:
                    inputs[0].backward(grad / inputs[1])
                if inputs[1].requires_grad:
                    inputs[1].backward(grad * (inputs[0] * Tensor.full(inputs[1].shape, -1.0, device=inputs[1].device) / (inputs[1] * inputs[1])))
            elif op == 'matmul':
                if inputs[0].requires_grad:
                    inputs[0].backward(grad @ inputs[1].T)
                if inputs[1].requires_grad:
                    inputs[1].backward(inputs[0].T @ grad)
            elif op == 'mse_loss':
                # d(MSE)/d(pred) = 2 * (pred - target) / n
                pred, target = inputs[0], inputs[1]
                if pred.requires_grad:
                    n = pred.size
                    grad_input = (pred - target) * Tensor.full(pred.shape, 2.0 / n, device=pred.device)
                    # For scalar loss, grad should be 1.0, so just use grad_input
                    pred.backward(grad_input)
            elif op == 'relu':
                # d(ReLU)/dx = 1 if x > 0 else 0
                x = inputs[0]
                if x.requires_grad:
                    # Create mask: 1 where x > 0, 0 elsewhere
                    x_data = x.tolist()
                    mask_data = [[1.0 if val > 0 else 0.0 for val in (row if isinstance(row, list) else [row])] 
                                 for row in (x_data if isinstance(x_data[0], list) else [x_data])]
                    if len(x.shape) == 1:
                        mask_data = mask_data[0]
                    mask = Tensor(mask_data, device=x.device)
                    x.backward(grad * mask)
            elif op == 'sigmoid':
                # d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
                x, output = inputs[0], inputs[1] if len(inputs) > 1 else None
                if x.requires_grad and output is not None:
                    grad_input = output * (Tensor.ones(output.shape, device=output.device) - output)
                    x.backward(grad * grad_input)
            elif op == 'tanh':
                # d(tanh)/dx = 1 - tanh(x)^2
                x, output = inputs[0], inputs[1] if len(inputs) > 1 else None
                if x.requires_grad and output is not None:
                    grad_input = Tensor.ones(output.shape, device=output.device) - (output * output)
                    x.backward(grad * grad_input)
            elif op == 'softmax':
                # d(softmax)/dx = softmax(x) * (1 - softmax(x)) for each element
                x, output, _ = inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None
                if x.requires_grad and output is not None:
                    # Create Jacobian-vector product for softmax gradient
                    # For simplicity, we compute element-wise gradient
                    grad_input = output * (Tensor.ones(output.shape, device=output.device) - output)
                    x.backward(grad * grad_input)
            elif op == 'transpose':
                # d(transpose)/dx = transpose(grad)
                x = inputs[0]
                if x.requires_grad:
                    x.backward(grad.T)
            elif op == 'sqrt':
                # d(sqrt(x))/dx = 1/(2*sqrt(x))
                x, output = inputs[0], inputs[1] if len(inputs) > 1 else None
                if x.requires_grad and output is not None:
                    grad_input = Tensor.full(output.shape, 0.5, device=output.device) / output
                    x.backward(grad * grad_input)
            elif op == 'pow':
                # d(x^p)/dx = p * x^(p-1)
                x, power = inputs[0], inputs[1]
                if x.requires_grad:
                    grad_input = Tensor.full(x.shape, power, device=x.device) * (x ** (power - 1))
                    x.backward(grad * grad_input)
            elif op == 'sum':
                # d(sum(x))/dx = grad broadcasted to x's shape
                x, dim = inputs[0], inputs[1]
                if x.requires_grad:
                    if dim is None:
                        grad_input = Tensor.full(x.shape, grad.tolist()[0], device=x.device)
                    else:
                        # Broadcast grad along specified dimension
                        grad_shape = list(x.shape)
                        grad_shape[dim] = 1
                        grad_reshaped = Tensor(grad.tolist(), shape=tuple(grad_shape), device=grad.device)
                        grad_input = grad_reshaped
                        for _ in range(x.ndim - grad_reshaped.ndim):
                            grad_input = grad_input.repeat_interleave(x.shape[dim], dim)
                    x.backward(grad_input)
            elif op == 'mean':
                # d(mean(x))/dx = grad broadcasted to x's shape divided by number of elements reduced
                x, dim = inputs[0], inputs[1]
                if x.requires_grad:
                    if dim is None:
                        n = x.size
                        grad_input = Tensor.full(x.shape, grad.tolist()[0] / n, device=x.device)
                    else:
                        n = x.shape[dim]
                        grad_shape = list(x.shape)
                        grad_shape[dim] = 1
                        grad_reshaped = Tensor(grad.tolist(), shape=tuple(grad_shape), device=grad.device)
                        grad_input = grad_reshaped
                        for _ in range(x.ndim - grad_reshaped.ndim):
                            grad_input = grad_input.repeat_interleave(x.shape[dim], dim)
                        grad_input = grad_input / n
                    x.backward(grad_input)
            elif op == 'exp':
                # d(exp(x))/dx = exp(x)
                x, output = inputs[0], inputs[1] if len(inputs) > 1 else None
                if x.requires_grad and output is not None:
                    x.backward(grad * output)
    
    def __iter__(self):
        """Make tensor iterable (iterate over first dimension)."""
        if len(self._shape) == 0:
            raise TypeError("iteration over a 0-d tensor")
        
        # Get full data as nested list
        data_list = self.tolist()
        
        # If 1D tensor, yield each element as a scalar
        if len(self._shape) == 1:
            for item in data_list:
                yield item
        else:
            # For higher dimensions, yield each row as a tensor
            for row in data_list:
                yield Tensor([row] if not isinstance(row, list) else row, 
                           device=self.device, dtype=self.dtype)

    def __len__(self):
        """Return length of first dimension."""
        if len(self._shape) == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._shape[0]
    
    def __eq__(self, other: object) -> bool:
        """Check equality of two tensors."""
        if isinstance(other, Tensor):
            if self._shape != other._shape or self.device != other.device or self.dtype != other.dtype:
                return False
            return self.tolist() == other.tolist()
        elif is_numpy_array(other):
            return self.tolist() == other.tolist()
        elif isinstance(other, list):
            return self.tolist() == other
        else:
            raise ValueError("Equality comparison only supported between Tensor objects")
    
    @property
    def T(self) -> 'Tensor':
        """Transpose the last two dimensions."""
        if len(self._shape) < 2:
            return self
        
        result = Tensor.__new__(Tensor)
        result._shape = self._shape[:-2] + (self._shape[-1], self._shape[-2])
        result._size = self._size
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.transpose(self._c_tensor)
        
        # Track gradient through transpose
        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('transpose', self)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def sqrt(self) -> 'Tensor':
        """Element-wise square root."""
        result = Tensor.__new__(Tensor)
        result._shape = self._shape
        result._size = self._size
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.sqrt(self._c_tensor)
        
        # Track gradient: d(sqrt(x))/dx = 1/(2*sqrt(x))
        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('sqrt', self, result)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result

    def sum(self, dim: Optional[int] = None) -> 'Tensor':
        """Sum along specified dimension."""
        if dim is not None and (dim < 0 or dim >= len(self._shape)):
            raise ValueError(f"Dimension out of range (expected to be in range of [0, {len(self._shape)-1}], but got {dim})")
        
        result = Tensor.__new__(Tensor)
        c_dim = -1 if dim is None else dim
        
        if dim is None:
            # Sum all elements
            result._shape = ()
            result._size = 1
        else:
            # Sum along specified dimension
            result_shape = list(self._shape)
            del result_shape[dim]
            result._shape = tuple(result_shape)
            result._size = Tensor._compute_size(result._shape)
        
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.sum(self._c_tensor, c_dim)
        
        # Track gradient through sum
        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('sum', self, dim)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result

    def mean(self, dim: Optional[int] = None) -> 'Tensor':
        """Mean along specified dimension."""
        if dim is not None and (dim < 0 or dim >= len(self._shape)):
            raise ValueError(f"Dimension out of range (expected to be in range of [0, {len(self._shape)-1}], but got {dim})")
        
        result = Tensor.__new__(Tensor)
        c_dim = -1 if dim is None else dim
        
        if dim is None:
            # Mean of all elements
            result._shape = ()
            result._size = 1
        else:
            # Mean along specified dimension
            result_shape = list(self._shape)
            del result_shape[dim]
            result._shape = tuple(result_shape)
            result._size = Tensor._compute_size(result._shape)
        
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.mean(self._c_tensor, c_dim)
        
        # Track gradient through mean
        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('mean', self, dim)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def log(self) -> 'Tensor':
        """Element-wise natural logarithm."""
        result = Tensor.__new__(Tensor)
        result._shape = self._shape
        result._size = self._size
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.log(self._c_tensor)
        
        # Track gradient: d(log(x))/dx = 1/x
        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('log', self)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result

    def exp(self) -> 'Tensor':
        """Element-wise exponential."""
        result = Tensor.__new__(Tensor)
        result._shape = self._shape
        result._size = self._size
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.exp(self._c_tensor)
        
        # Track gradient: d(exp(x))/dx = exp(x)
        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('exp', self)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result