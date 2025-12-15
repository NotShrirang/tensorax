"""
Type checking utilities.

This module provides functions to check the types of various objects,
such as whether an object is a Tensor, a NumPy array, or other types.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorax.tensor import Tensor

def is_tensor(x):
    return isinstance(x, Tensor)

def is_numpy_array(x):
    return type(x).__module__ == "numpy" and type(x).__name__ == "ndarray"