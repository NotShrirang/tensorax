"""
Tensora - High-performance tensor library with CUDA acceleration
"""

__version__ = '0.1.0'

from .tensor import Tensor
from . import nn
from . import optim
from . import functional as F

__all__ = [
    'Tensor',
    'nn',
    'optim',
    'F',
]

# Check if CUDA extension is available
try:
    from . import _C
    _cuda_available = hasattr(_C, 'cuda_is_available') and _C.cuda_is_available()
except ImportError:
    _cuda_available = False 

def cuda_is_available():
    """Check if CUDA is available."""
    return _cuda_available
