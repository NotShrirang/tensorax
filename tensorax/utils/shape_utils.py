from typing import Union, Tuple, Optional, TYPE_CHECKING
import warnings

try:
    from tensorax import _C
except ImportError as e:
    warnings.warn(f"Failed to import _C module: {e}. Tensor operations will not work until the package is built.")
    _C = None  # Will be available after building

import tensorax as ts
from tensorax.utils.type_checks import is_numpy_array

def _compute_size(shape: Tuple[int, ...]) -> int:
    """Compute total number of elements from shape."""
    size = 1
    for dim in shape:
        size *= dim
    return size

def _has_valid_shape(data: Union[list, 'ts.Tensor']) -> bool:
    """Check if the nested list has a valid shape (all sublists have the same length)."""
    if is_numpy_array(data):
        return True  # NumPy arrays are assumed to have valid shapes

    if isinstance(data, ts.Tensor):
        return True  # Tensors are assumed to have valid shapes

    if not data:
        return True  # An empty list is considered to have a valid shape
    
    if isinstance(data, (int, float)):
        return True  # A single number is valid
    
    if len(data) == 0:
        return True
    
    if isinstance(data[0], (int, float)):
        for item in data:
            if not isinstance(item, (int, float)):
                return False
    elif isinstance(data[0], (list, 'ts.Tensor', tuple, set)):
        expected_length = len(data[0])
        for item in data:
            if not isinstance(item, (list, 'ts.Tensor', tuple, set)) or len(item) != expected_length:
                return False
            if not _has_valid_shape(item):
                return False

    return True

def _infer_shape(data: Union[list, 'ts.Tensor']) -> Tuple[int, ...]:
    """Infer the shape of a nested list or Tensor."""
    shape = []
    current_level = data

    while isinstance(current_level, list):
        shape.append(len(current_level))
        if len(current_level) == 0:
            break
        current_level = current_level[0]

    return tuple(shape)
