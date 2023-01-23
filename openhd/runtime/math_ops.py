"""
=====================
Math Operators
=====================

This implements math operations compiled in AOT as runtime library
(e.g., cos, sign) 
"""

import numpy as np

from ..dev import debug

from .. import core
from ..core.hypervector import hypervector, hypermatrix
from ..core.array import array

from ..backend import cuda_impl as ci

TAG = "runtime.math_ops"

def prepare_math_op_module():
    # Load the source to compile in AOT with the header
    cuda_code = ci.append_cuda_header(
            core.__OPENHD_MODULE_PATH__ + "/runtime/cu/math_ops.cu",
            {'__D__': core._D})

    # Load module (compile if needed)
    cubin_filename = ci.load_module_from_code_str(cuda_code)
    mod = ci.get_module(cubin_filename)

    # Prepare the functions for each API
    cos.cu_func = mod.get_function("cosine")
    sign.cu_func = mod.get_function("sign")
    fill.cu_func = mod.get_function("fill")

def cos(hv_matrix):
    """
    Perform in-place cosine operation

    Args:
        hypermatrix: A hypermatrix 

    Return:
        hypermatrix: The same hypermatrix converted
    """

    # Run CUDA kernel
    __MAX_THREADS__ = ci.DeviceData().max_threads
    cos.cu_func(
            hv_matrix.mem, 
            block=(__MAX_THREADS__, 1, 1),
            grid=(
                (core._D + __MAX_THREADS__ - 1) // __MAX_THREADS__,
                hv_matrix.N)
            )

    return hv_matrix 

def sign(hv_matrix):
    """
    Perform in-place sign operation,
    which takes -1 and +1 based on a stored value

    Args:
        hypermatrix: A hypermatrix 

    Return:
        hypermatrix: The same hypermatrix converted
    """

    # Run CUDA kernel
    __MAX_THREADS__ = ci.DeviceData().max_threads
    sign.cu_func(
            hv_matrix.mem, 
            block=(__MAX_THREADS__, 1, 1),
            grid=(
                (core._D + __MAX_THREADS__ - 1) // __MAX_THREADS__,
                hv_matrix.N)
            )

    return hv_matrix 

def fill(hv_matrix, value):
    """
    Perform in-place value filling operation,

    Args:
        hypermatrix: A hypermatrix 

    Return:
        hypermatrix: The hypermatrix filled with value
    """

    # Run CUDA kernel
    __MAX_THREADS__ = ci.DeviceData().max_threads
    fill.cu_func(
            hv_matrix.mem, np.float32(value), 
            block=(__MAX_THREADS__, 1, 1),
            grid=(
                (core._D + __MAX_THREADS__ - 1) // __MAX_THREADS__,
                hv_matrix.N)
            )

    return hv_matrix
