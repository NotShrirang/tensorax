import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("Error: pybind11 is required. Install it with: pip install pybind11")
    sys.exit(1)


# Read version from __init__.py
def read_version():
    with open(os.path.join('tensorax', '__init__.py'), 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'


# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


def find_cuda():
    """Find CUDA installation"""
    # Check environment variable
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    
    if cuda_home and os.path.exists(cuda_home):
        return cuda_home
    
    # Try common installation paths
    common_paths = [
        '/usr/local/cuda',
        '/opt/cuda',
        '/usr/lib/cuda',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # Try finding nvcc in PATH
    try:
        nvcc_path = subprocess.check_output(['which', 'nvcc'], 
                                           stderr=subprocess.DEVNULL).decode().strip()
        if nvcc_path:
            # nvcc is typically in /path/to/cuda/bin/nvcc
            cuda_home = str(Path(nvcc_path).parent.parent)
            if os.path.exists(cuda_home):
                return cuda_home
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return None


class CUDAExtension(Extension):
    """Custom Extension class for CUDA compilation"""
    pass


class BuildExtension(build_ext):
    """Custom build_ext to compile CUDA code"""
    
    def build_extensions(self):
        # Allow .cu files
        self.compiler.src_extensions.append('.cu')
        
        # Store original compile method
        original_compile = self.compiler._compile
        original_compiler_so = None
        
        def custom_compile(obj, src, ext_type, cc_args, extra_postargs, pp_opts):
            nonlocal original_compiler_so
            
            # Save original compiler on first call
            if original_compiler_so is None:
                original_compiler_so = self.compiler.compiler_so[:]
            
            if src.endswith('.cu'):
                # Use nvcc for .cu files
                self.compiler.set_executable('compiler_so', 'nvcc')
                # Get CUDA-specific flags
                if isinstance(extra_postargs, dict):
                    extra_postargs = extra_postargs.get('nvcc', [])
            else:
                # Use default compiler for .cpp files
                self.compiler.compiler_so = original_compiler_so[:]
                if isinstance(extra_postargs, dict):
                    extra_postargs = extra_postargs.get('cxx', [])
            
            return original_compile(obj, src, ext_type, cc_args, extra_postargs, pp_opts)
        
        # Replace compile method
        self.compiler._compile = custom_compile
        
        # Customize compiler flags per extension
        for ext in self.extensions:
            if isinstance(ext, CUDAExtension):
                # CUDA-specific flags
                ext.extra_compile_args = {
                    'cxx': ['-O3', '-std=c++17', '-fPIC'],
                    'nvcc': [
                        '-O3',
                        '--use_fast_math',
                        '-std=c++17',
                        '--compiler-options', '-fPIC',
                        '-gencode=arch=compute_75,code=sm_75',  # T4, RTX 20xx
                        '-gencode=arch=compute_80,code=sm_80',  # A100
                        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
                        '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx (CUDA 11.8+)
                    ]
                }
            else:
                # CPU-only flags
                ext.extra_compile_args = ['-O3', '-std=c++17', '-fPIC']
        
        build_ext.build_extensions(self)


# CUDA extension modules
ext_modules = []

# Check if CUDA is available
cuda_home = find_cuda()
cuda_available = cuda_home is not None

if cuda_available:
    print(f"Found CUDA at: {cuda_home}")
    print("Building with CUDA support")
    
    # CUDA extensions
    cuda_extension = CUDAExtension(
        name='tensorax._C',
        sources=[
            'csrc/tensor_ops.cpp',
            'csrc/cpu/tensor_cpu.cpp',  # CPU implementations needed too!
            'csrc/cuda/tensor_cuda.cu',
            'csrc/cuda/kernels/elementwise.cu',
            'csrc/cuda/kernels/reduction.cu',
            'csrc/cuda/kernels/matmul.cu',
        ],
        include_dirs=[
            'csrc',
            'csrc/cuda',
            pybind11_include,
            os.path.join(cuda_home, 'include'),
        ],
        library_dirs=[
            os.path.join(cuda_home, 'lib64'),
            os.path.join(cuda_home, 'lib'),
        ],
        libraries=['cudart'],
        define_macros=[('WITH_CUDA', None)],
        language='c++',
    )
    ext_modules.append(cuda_extension)
else:
    print("CUDA not found. Building CPU-only version")
    
    # CPU-only extension
    cpu_extension = Extension(
        name='tensorax._C',
        sources=[
            'csrc/tensor_ops.cpp',
            'csrc/cpu/tensor_cpu.cpp',
        ],
        include_dirs=[
            'csrc',
            pybind11_include,
        ],
        language='c++',
    )
    ext_modules.append(cpu_extension)

setup(
    name='tensorax',
    version=read_version(),
    author='Shrirang Mahajan',
    author_email='shrirangmahajan123@gmail.com',
    description='A high-performance tensor library with CUDA acceleration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NotShrirang/tensorax',
    packages=find_packages(exclude=['tests', 'examples', 'docs']) + ['tensorax.utils'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        # Pure C++/CUDA backend - minimal dependencies!
        'pybind11>=2.6.0',  # Python-C++ bindings
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.900',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensor deep-learning cuda gpu machine-learning',
)
