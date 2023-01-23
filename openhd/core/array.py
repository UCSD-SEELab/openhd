"""
============
OpenHD Array
============

The primitive data types: Array

seealso: memory.array_pool
"""
import numpy as np

from ..dev import debug

from ..memory import array_pool

from ..backend import cuda_impl as ci

TAG = "core.openhd_array"

class array:
    """
    OpenHD Array Type
    """
    def __init__(self, shape, dtype=np.int32):
        """
        Allocate array memory on GPU side
        """
        assert(dtype == np.int32 or dtype==np.float32)

        self.shape = shape
        if isinstance(shape, tuple):
            debug.thrifty_assert(
                    len(shape) == 2,
                    "The OpenHD array shape has to be 2D")
            self.size = shape[0] * shape[1]
        else:
            self.size = shape

        self.dtype = dtype
        self.mem = array_pool.alloc(self.size)

    def __del__(self):
        """
        Called due to the reference counter
        """
        array_pool.free(self.mem)

    def to_numpy(self):
        """
        Convert to a numpy array
        """
        nparray = np.empty(self.shape, dtype=self.dtype)
        ci.drv.memcpy_dtoh(nparray, self.mem)
        return nparray

    def from_numpy(self, ndarray):
        """
        Copy data from a numpy array
        """
        ci.drv.memcpy_htod(self.mem, ndarray)
        return self


    def get_gpu_mem_structure(self):
        stride = np.int32(0)
        if isinstance(self.shape, tuple):
            stride = np.int32(self.shape[0])

        return self.mem, stride 

    def __getitem__(self, idx):
        debug.log(
                debug.WARNING, TAG,
                "Accessing elements of OpenHD array is not recommended. " + \
                        "Use to_numpy() instead.",
                        print_once=True)

        nparray = self.to_numpy()
        return nparray[idx]

    def __repr__(self):
        return "<OpenHD Array type, shape=%s, dtype=%s>" % (
                str(self.shape), str(self.dtype))

