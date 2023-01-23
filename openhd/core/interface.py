"""
=================
Interface library
=================

This includes the core functions for users' python implementations, e.g., run
"""
import os
import numpy as np
import inspect
import hashlib

from ..dev import debug

from ..core import hypervector
from ..core import array

from ..jit import jit_compiler
from ..jit import HYPERVECTOR_TYPE, HYPERMATRIX_TYPE
from ..jit import NP_INT_ARRAY_TYPE, NP_FLOAT_ARRAY_TYPE
from ..jit import HD_INT_ARRAY_TYPE, HD_FLOAT_ARRAY_TYPE
from ..jit import PREPROC_FEATURES, prebuilt_encode_variables
from ..jit.encode_merger import merge_encode_function

from ..backend.variable_type_array import VariableTypeArray
from ..backend import cuda_impl as ci

from ..runtime.assoc_ops import prepare_assoc_op_module
from ..runtime.math_ops import prepare_math_op_module

from ..memory.array_pool import ArrayPoolManager

from .. import core as core

# Modules to run the jit code ##################
import openhd as __hd__
import openhd.backend.cuda_impl as __ci__
import time as __time__
import numpy as __np__
################################################


TAG = "core.interface"

JIT_CACHE_PATH = ".openhd"

def init(
        D=10000, context=None, HD_ARG_BUF_SIZE=128, N_STREAM=2,
        POOLED_ARRAY_CNT=4):
    """
    Initialize the OpenHD framework

    Note: To use the global variable, pass context with globals()
    """
    # This will be accessed by jit-compiled python
    global __D__, __HD_ARG_ARRAY__, __SHUFFLE_INDICES_CPU__

    # Dimension
    core.__dict__["_D"] = D
    __D__ = D

    # JIT path
    module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    core.__dict__["__OPENHD_MODULE_PATH__"] = module_path

    cache_path = os.path.expanduser("~/") + JIT_CACHE_PATH
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    core.__dict__["__JIT_CACHE_PATH__"] = cache_path + "/"

    # Memory management
    __HD_ARG_ARRAY__ = VariableTypeArray(HD_ARG_BUF_SIZE)
    core.__dict__["__ARRAY_POOL_MANAGER__"] = ArrayPoolManager(POOLED_ARRAY_CNT)

    # Context
    if context is None:
        context = globals()
    core.__dict__["__CONTEXT__"] = context

    # Stream count
    core.__dict__["__N_STREAM__"] = N_STREAM

    # Create shuffle indices
    if __D__ < 65535:
        item_type = np.ushort
    else:
        item_type = np.uint32

    core.__dict__["__SHUFFLE_INDICES_CPU__"] = \
            np.random.permutation(__D__).astype(item_type)
    __SHUFFLE_INDICES_CPU__ = core.__dict__["__SHUFFLE_INDICES_CPU__"]

    # Load the runtime
    prepare_assoc_op_module()
    prepare_math_op_module()

def run(func):
    """
    (Decorator) Run a function as JIT-compiled cuda code

    Example:
        >>> @hd.run()
            def foo(...):
                ... Implementations using OpenHD ...
    """

    def wrapper(*args, **kwargs):
        result = _run_body(func, args, kwargs)
        return result

    return wrapper


def encode(
        encode_function, feature_matrix,
        extra_args=None, preprocess_function=None):
    """
    Encode the given feature matrix to hypervector

    Args:
        encode_function: a function that encodes the features of a sample
        def encode_function(
            input_features, output_hypervector, extra_arg1, extra_arg2, ...):

        feature_matrix: Feature matrix (numpy array)

        extra_args: extra argument passed to the given functions

        preprocess_function: a function that preprocess a feature
        def preprocess_function(
            original_feature, preprocessed_feature, extra_arg1, ...):
    """

    _check_encode_function_args(
            encode_function, extra_args, preprocess_function)

    debug.log(debug.INFO, TAG, "Compiling encode: " + str(encode_function))
    func_name = _compile_encode(
            encode_function, extra_args, feature_matrix, preprocess_function)

    args = [feature_matrix, None] + list(extra_args)

    ret = globals()[func_name](*args)
    return ret


def default_preprocssor(original_feature, preprocessed_feature):
    preprocessed_feature = original_feature # Copy all of them

def _check_encode_function_args(
        encode_function, extra_args, preprocess_function):
    # Sanity check of the arguments
    n_extra_args = 0
    if extra_args is not None:
        n_extra_args = len(extra_args)

    n_encoder_args = len(inspect.getargspec(encode_function).args)
    debug.thrifty_assert(
            n_extra_args + 2 == n_encoder_args,
            "The arguments of the encode function " + \
                    "can not be interpreted to the desired format."
            )

    if preprocess_function is not None:
        n_preprocessor_args = len(inspect.getargspec(
            preprocess_function).args)
        debug.thrifty_assert(
                n_extra_args + 2 == n_encoder_args,
                "The arguments of the preprocessor function " + \
                        "can not be interpreted to the desired format."
                )



def _run_body(func, args, kwargs):
    """
    Run the function on CUDA with JIT compile (if needed)
    """
    debug.log(debug.INFO, TAG, "Compiling: " + str(func))
    func_name = _jit_compile(func, args, kwargs)

    ret = globals()[func_name](*args)

    return ret


def get_item_type(val):
    if isinstance(val, hypervector.hypervector):
        item_type = HYPERVECTOR_TYPE
    elif isinstance(val, hypervector.hypermatrix):
        item_type = HYPERMATRIX_TYPE
    elif isinstance(val, np.ndarray):
        if val.dtype.type == np.int64 or val.dtype.type == np.int32:
            item_type = NP_INT_ARRAY_TYPE
        elif val.dtype.type == np.float64 or val.dtype.type == np.float32:
            item_type = NP_FLOAT_ARRAY_TYPE
        else:
            debug.thrifty_assert(False,
                    "Numpy array should be float or int")
    elif isinstance(val, array.array):
        if val.dtype.type == np.int32:
            item_type = HD_INT_ARRAY_TYPE
        elif val.dtype.type == np.float32:
            item_type = HD_FLOAT_ARRAY_TYPE
        else:
            debug.thrifty_assert(False,
                    "Numpy array should be float or int")
    else:
        item_type = type(val)

    return item_type


def _jit_compile(func, args, kwargs):
    """
    Perfrom JIT compile
    """

    # Obtain code string of the function body
    py_source = inspect.getsource(func)
    debug.log(debug.DEV, TAG, py_source, add_lineno=True)

    # Create argument types 
    debug.thrifty_assert(len(kwargs) == 0, "TODO: Kwargs is not supported yet")

    arg_variables = dict()
    for name, val in zip(inspect.getargspec(func).args, args):
        item_type = get_item_type(val)
        arg_variables[name] = (item_type, val)

    # Try to find in cache
    filename = _gen_pycode_cache_filename(py_source, arg_variables)

    jit_py_code = _try_read_cached_pycode(filename)

    if jit_py_code is None:
        # Run JIT compiler
        jit_py_code, cubin_list = jit_compiler.compile(py_source, arg_variables)
        _cache_pycode(filename, jit_py_code, cubin_list)

    debug.log(debug.DEV, TAG, jit_py_code, add_lineno=True)
    #assert(False)

    exec(jit_py_code, globals())
    func_name = func.__name__

    return func_name

def _gen_pycode_cache_filename(py_source, arg_variables):
    # TODO: Also check the global variable 
    return 'jit_' + str(
            hashlib.md5(str(py_source + str(arg_variables)).encode('utf-8')).hexdigest())

def _try_read_cached_pycode(filename):
    """
    Try to read cached python code
    """
    # TODO: Read the caches in the init
    py_filename = core.__JIT_CACHE_PATH__ + filename + ".py"
    cubinlist_filename = core.__JIT_CACHE_PATH__ + filename + ".cubin_list"

    if os.path.exists(py_filename):
        debug.log(debug.INFO, TAG, "Cache Found: " + str(py_filename))
    else:
        return None

    with open(py_filename) as infile:
        jit_py_code = infile.read()

    with open(cubinlist_filename) as infile:
        for cubin_filename in infile:
            ci.load_module(cubin_filename.strip())

    return jit_py_code


def _cache_pycode(filename, jit_py_code, cubin_list):
    py_filename = core.__JIT_CACHE_PATH__ + filename + ".py"
    cubinlist_filename = core.__JIT_CACHE_PATH__ + filename + ".cubin_list"

    with open(py_filename, "w") as out:
        out.write(jit_py_code)

    with open(cubinlist_filename, "w") as out:
        for cubin_filename in cubin_list:
            out.write(cubin_filename + "\n")

def _compile_encode(
        encode_function, extra_args, feature_matrix, preprocess_function):
    if preprocess_function is None:
        preprocess_function = default_preprocssor

    if not isinstance(feature_matrix, np.ndarray) or \
            feature_matrix.dtype.type != np.float32:
        feature_matrix = np.array(feature_matrix, dtype=np.float32)

    # Obtain code string of the function body
    str_encode_func = inspect.getsource(encode_function)
    encode_func_vars = inspect.getargspec(encode_function).args
    #debug.log(debug.DEV, TAG, str_encode_func, add_lineno=True)

    preproc_func_vars = inspect.getargspec(preprocess_function).args
    str_preprocess_func = inspect.getsource(preprocess_function)
    #debug.log(debug.DEV, TAG, str_encode_func, add_lineno=True)

    # Get the arguments
    arg_variables = dict()
    arg_variables[encode_func_vars[0]] = (NP_FLOAT_ARRAY_TYPE, 0)
    arg_variables[encode_func_vars[1]] = (HYPERMATRIX_TYPE, None)
    for name, val in zip(encode_func_vars[2:], extra_args):
        item_type = get_item_type(val)
        arg_variables[name] = (item_type, val)

    # Get memory size to determine stream size = Memory / Feature size / STREAM   
    M = ci.get_constant_memory_size(devicenum=0)                            
    M = int(M / (feature_matrix.shape[1] * 4) / core.__N_STREAM__) 
    debug.thrifty_assert(M > 0, "TODO: Not enough constant memory to stream")

    # Add predefined encoding variables
    arg_variables[PREPROC_FEATURES] = (NP_FLOAT_ARRAY_TYPE, 0)
    for var in prebuilt_encode_variables:
        arg_variables[var] = (int, 0)
    arg_variables["__F__"] = (int, feature_matrix.shape[1])
    arg_variables["__M__"] = (int, 1) # Will be updated in CUDA generation
    arg_variables["__N__"] = (int, feature_matrix.shape[0])
    arg_variables["__D__"] = (int, core._D)

    # Try to find in cache
    filename = _gen_pycode_cache_filename(
            str_preprocess_func + str_encode_func, arg_variables)
    jit_py_code = _try_read_cached_pycode(filename)

    if jit_py_code is None:
        # Merge with the preprocssor
        merged_py_code = merge_encode_function(
                str_preprocess_func, preproc_func_vars,
                str_encode_func, encode_func_vars)

        # Run JIT compiler
        jit_py_code, cubin_list = jit_compiler.compile(
                merged_py_code, arg_variables, encode_func_vars)
        _cache_pycode(filename, jit_py_code, cubin_list)

    debug.log(debug.DEV, TAG, jit_py_code, add_lineno=True)

    exec(jit_py_code, globals())
    func_name = encode_function.__name__
    return func_name


