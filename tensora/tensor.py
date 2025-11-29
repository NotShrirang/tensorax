"""
Core Tensor class for Tensora library.
Pure C++/CUDA backend - no NumPy dependency.
"""

from typing import Union, Tuple, Optional, List
try:
    from . import _C
except ImportError:
    _C = None  # Will be available after building

from tensora.utils.type_checks import is_numpy_array

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
    ):
        """
        Initialize a Tensor.
        
        Args:
            data: Input data (list, flat buffer, or another Tensor)
            shape: Shape of the tensor (inferred from list if not provided)
            dtype: Data type ('float32', 'float64', 'int32', 'int64')
            device: Device to place tensor on ('cpu' or 'cuda')
            requires_grad: Whether to track gradients
        """
        self.dtype = dtype or 'float32'
        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        
        if isinstance(data, Tensor):
            # Copy from another tensor
            self._shape = data._shape
            self._size = data._size
            self._c_tensor = _C.copy_tensor(data._c_tensor) if _C else None
        else:
            # Create from list/data
            flat_data, inferred_shape = self._flatten_data(data)
            self._shape = shape or inferred_shape
            self._size = self._compute_size(self._shape)
            
            # Create C++ tensor
            if _C:
                if device == 'cpu':
                    self._c_tensor = _C.create_tensor_cpu(flat_data, list(self._shape), self.dtype)
                else:
                    self._c_tensor = _C.create_tensor_cuda(flat_data, list(self._shape), self.dtype)
            else:
                self._c_tensor = None
                self._data = flat_data  # Fallback for testing before build
    
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
        size = 1
        for dim in shape:
            size *= dim
        return size
    
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
    
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
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
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        if self.device != other.device:
            raise RuntimeError(f"Tensors on different devices: {self.device} vs {other.device}")
        
        if len(self._shape) < 2 or len(other._shape) < 2:
            raise RuntimeError(f"Matrix multiplication requires 2D+ tensors, got {self._shape} and {other._shape}")
        
        if self._shape[-1] != other._shape[-2]:
            raise RuntimeError(f"Incompatible shapes for matmul: {self._shape} and {other._shape}. {self._shape[-1]} != {other._shape[-2]}")
        # Result shape: (..., M, K) @ (..., K, N) -> (..., M, N)
        result_shape = self._shape[:-1] + (other._shape[-1],)
        
        result = Tensor.__new__(Tensor)
        result._shape = result_shape
        result._size = self._compute_size(result_shape)
        result.dtype = self.dtype
        result.device = self.device
        result.grad = None
        
        if _C:
            result._c_tensor = _C.matmul(self._c_tensor, other._c_tensor)
        
        if self.requires_grad or other.requires_grad:
            result.requires_grad = True
            result._grad_fn = ('matmul', self, other)
        else:
            result.requires_grad = False
            result._grad_fn = None
        
        return result
    
    def __repr__(self) -> str:
        data_repr = self.tolist() if _C else "<not built>"
        return f"Tensor({data_repr}, shape={self.shape}, device='{self.device}', dtype='{self.dtype}')"
    
    def __str__(self) -> str:
        data_repr = self.tolist() if _C else "<not built>"
        return f"Tensor({data_repr})"
    
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
                    inputs[0].backward(grad)
                if inputs[1].requires_grad:
                    inputs[1].backward(grad)
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
