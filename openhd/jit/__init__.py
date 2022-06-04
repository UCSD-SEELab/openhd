"""
==============
JIT for OpenHD
==============
"""

# Built-in OpenHD types
HYPERVECTOR_TYPE = "hypervec_type"
HYPERMATRIX_TYPE = "hypermat_type"

HYPERELEMNT_TYPE = "hyperelt_type"

# Builtin array types (core.array)
HD_INT_ARRAY_TYPE = "hd_int_array_type"
HD_FLOAT_ARRAY_TYPE = "hd_float_array_type"

# Numpy type - we support it on CUDA for readonly 
NP_INT_ARRAY_TYPE = "np_int_array_type"
NP_FLOAT_ARRAY_TYPE = "np_float_array_type"

# py14 compatibility
ARG_PREFIX = "__ARG__"
STRIDE_POSTFIX = "__STRIDE__"


# encode built-in vars
FEATURE_STREAM = "__feature_stream__" # Constant memory of streaming
PREPROC_FEATURES = "__shared_features__"
prebuilt_encode_variables = [ # See encode_merger
        "__blockIdx_x__",
        "__F__",
        "__M__",
        "__N__",
        "__threadIdx_x__",
        "__blockDim_x__",
        "__blockIdx_y__",
        "__stream__",
        "__base_n__",
        ]

# Cuda built-in vars
cuda_builtin_vars = {
        "__blockIdx_x__" : "blockIdx.x",
        "__threadIdx_x__" : "threadIdx.x",
        "__blockDim_x__" : "blockDim.x",
        "__blockIdx_y__": "blockIdx.y",
        }


# shuffle built-in vars
SHUFFLE_INDICES = "__shuffle_indices__" # Constant memory for indices

# Transpiled kernel function
TRANSPILED_KERNEL_FUNCTION = "__openhd_transpiled_kernel_function__"
