"""
=====================
Associative Operators
=====================

This implements associtative operations compiled in AOT as runtime library
(e.g., search, cossim) 
"""

import numpy as np
from skcuda import cublas as cb

from ..dev import debug

from .. import core
from ..core.hypervector import hypervector, hypermatrix
from ..core.array import array

from ..backend import cuda_impl as ci

TAG = "runtime.assoc_ops"

def prepare_assoc_op_module():
    # Load the source to compile in AOT with the header
    cuda_code = ci.append_cuda_header(
            core.__OPENHD_MODULE_PATH__ + "/runtime/cu/assoc_ops.cu",
            {'__D__': core._D})

    # Load module (compile if needed)
    cubin_filename = ci.load_module_from_code_str(cuda_code)
    mod = ci.get_module(cubin_filename)

    # Prepare the functions for each API
    search.cu_func = mod.get_function("search")
    cossim.cu_func = mod.get_function("cossim")

def check_and_return_input_dims(hv_matrix, query):
    debug.thrifty_assert(
            isinstance(hv_matrix, hypermatrix),
            "query has to be hypervector or hypermatrix.")
    N_am = hv_matrix.N # number of associative mem

    if isinstance(query, hypervector):
        N_query = 1
    else: # query 
        debug.thrifty_assert(
                isinstance(query, hypermatrix),
                "query has to be hypervector or hypermatrix.")
        N_query = query.N
        
    return N_am, N_query

def search(hv_matrix, query):
    """
    Perform search operation

    Args:
        hypermatrix: The hypermatrix searched 
        query: Query hypervector or hypermatrix

    Return:
        Index - Integer if query is hypervector, or 
        Indices - OpenHD Integer Array if query is hypermatrix
    """
    # Create the result array
    N_am, N_query = check_and_return_input_dims(hv_matrix, query)
    result_array = array(N_query, dtype=np.int32)

    # Run CUDA kernel
    __MAX_THREADS__ = ci.DeviceData().max_threads
    search.cu_func(
            hv_matrix.mem, query.mem,
            result_array.mem,
            np.int32(N_am), np.int32(N_query),
            block=(__MAX_THREADS__, 1, 1),
            grid=((N_query + __MAX_THREADS__ - 1) // __MAX_THREADS__, 1))

    return result_array 

def cossim(hv_matrix, query):
    """
    Compute cosine similarity

    Args:
        hypermatrix: The hypermatrix computed 
        query: Query hypervector or hypermatrix

    Return:
        1D OpenHD float array if query is hypervector, or 
        2D OpenHD float array if query is hypermatrix
    """
    # Create the result array
    N_am, N_query = check_and_return_input_dims(hv_matrix, query)
    shape = (N_am, N_query)
    result_array = array(shape, dtype=np.float32)

    # Run CUDA kernel
    __MAX_THREADS__ = ci.DeviceData().max_threads
    cossim.cu_func(
            hv_matrix.mem, query.mem,
            result_array.mem,
            np.int32(N_am), np.int32(N_query),
            block=(__MAX_THREADS__, 1, 1),
            grid=((N_query + __MAX_THREADS__ - 1) // __MAX_THREADS__, 1))

    return result_array

def matmul(ndarray, hv_matrix):
    """
    Matrix multiplication with HV matrix. Mainly used for non-linear encoding

    Args:
        ndarray: Numpy 2D array
        hv_matrix: hv matrix

    Return:
        Hypermatrix type
    """
    # Check and copy array to GPU
    debug.thrifty_assert(
            isinstance(ndarray, np.ndarray),
            "ndarray has to be 2D numpy array")
    debug.thrifty_assert(
            len(ndarray.shape) == 2,
            "ndarray has to be 2D array")

    gpuarray = array(ndarray.shape, dtype=np.float32).from_numpy(ndarray)

    # Create the result hypermatrix
    N, F = ndarray.shape
    hvmat = hypermatrix(N) 

    h = cb.cublasCreate() # initialize cublas context
    cb.cublasSgemm(h, 'n', 'n',
            core._D, N, F,
            np.float(1.0),
            hv_matrix.mem, core._D,
            gpuarray.mem, F,
            np.float(1.0),
            hvmat.mem, core._D)
    cb.cublasDestroy(h)

    return hvmat


