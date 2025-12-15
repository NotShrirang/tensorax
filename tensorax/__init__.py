"""
Tensorax - High-performance tensor library with CUDA acceleration
"""

__version__ = '0.1.0'

from .tensor import Tensor
from . import nn
from . import optim
from . import functional as F
from .constants import *

__all__ = [
    # Core classes and modules
    'Tensor',
    'nn',
    'optim',
    'F',

    # Data types
    'int8',
    'int16',
    'int32',
    'int64',

    'uint8',
    'uint16',
    'uint32',
    'uint64',

    'float16',
    'float32',
    'float64',

    'complex64',
    'complex128',

    # Devices
    'cpu',
    'cuda',
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
