"""
===================
Hypervector classes
===================

The primitive data types: Hypervectors and Hypermatrix
"""
import numpy as np

from ..dev import debug

from .. import core
from ..backend import cuda_impl as ci

TAG = "core.hypervector"

def get_range_of_supported_dtype(dtype):
    supported_types = [
            np.bool,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.float32,
            ]

    debug.thrifty_assert(
            dtype in supported_types,
            "The type %s is not supported " % str(dtype) + \
                    "as the element type of hypervector in OpenHD." 
            )

    if dtype is np.bool:
        return -1, 1

    if dtype is np.float32:
        return np.finfo(dtype).min, np.finfo(dtype).max

    return np.iinfo(dtype).min, np.iinfo(dtype).max


class hypervector:
    """
    Single hypervector structure
    """
    def __init__(self, dtype=np.float32):
        self.mem = None
        self.dtype = dtype
        self.elem_range = get_range_of_supported_dtype(self.dtype)
        self.alloc()

    def alloc(self):
        """
        Allocate memory on GPU side
        """
        self.mem = ci.drv.mem_alloc(core._D * np.dtype(self.dtype).itemsize)
        debug.log(debug.DEV, TAG, "Created Hypervector: " + \
                str(hex(id(self.mem))))

    def set_elem_range(self, elem_range):
        assert(self.elem_range[0] <= elem_range[0] and \
                self.elem_range[1] >= elem_range[1])
        self.elem_range = elem_range

    def debug_print_values(self):
        hv = np.empty(core._D, dtype=self.dtype)
        ci.drv.memcpy_dtoh(hv, self.mem)
        debug.log(debug.DEV, TAG, '\n' + str(hv))

    def to_numpy(self):
        nparray = np.empty(core._D, dtype=self.dtype)
        ci.drv.memcpy_dtoh(nparray, self.mem)
        return nparray

    def from_numpy(self, ndarray):
        assert(ndarray.shape == (core._D,))
        ci.drv.memcpy_htod(self.mem, ndarray.astype(self.dtype))
        return self

    def __del__(self):
        debug.log(debug.DEV, TAG, "Deleted Hypervector: " + \
                str(hex(id(self.mem))))

    def __repr__(self):
        return "<hypervector type, D=%d>" % core._D

class hypermatrix:
    """
    Hypervector matrix structure
    """
    def __init__(self, N, dtype=np.float32):
        self.N = N
        self.mem = None
        self.dtype = dtype
        self.elem_range = get_range_of_supported_dtype(self.dtype)
        self.alloc()

    def alloc(self):
        """
        Allocate memory on GPU side
        """
        self.mem = ci.drv.mem_alloc(
                self.N * core._D * np.dtype(self.dtype).itemsize)
        debug.log(debug.DEV, TAG, "Created Hypervector: " + \
                str(hex(id(self.mem))))

    def set_elem_range(self, elem_range):
        assert(self.elem_range[0] <= elem_range[0] and \
                self.elem_range[1] >= elem_range[1])
        self.elem_range = elem_range

    def debug_print_values(self):
        hv = np.empty((self.N, core._D), dtype=self.dtype)
        ci.drv.memcpy_dtoh(hv, self.mem)
        debug.log(debug.DEV, TAG, '\n' + str(hv))

    def to_numpy(self):
        nparray = np.empty((self.N, core._D), dtype=self.dtype)
        ci.drv.memcpy_dtoh(nparray, self.mem)
        return nparray

    def from_numpy(self, ndarray):
        assert(ndarray.shape == (self.N, core._D))
        ci.drv.memcpy_htod(self.mem, ndarray.reshape(-1).astype(self.dtype))
        return self

    def __del__(self):
        debug.log(debug.DEV, TAG, "Deleted Hypervector: " + \
                str(hex(id(self.mem))))

    def __repr__(self):
        return "<hypermatrix type, D=%d, N=%d>" % (core._D, self.N)

