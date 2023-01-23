"""
===================
Backend with PyCuda
===================

This includes backend implemenations using PyCuda

TODO: Better documnetation in development 
"""
import os
import hashlib
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData

from ..dev import debug

from .. import core

TAG = "backend.cuda_impl"

def load_module(filename):
    """
    Load cubin file as a module
    """
    if filename not in load_module.cuda_modules:
        load_module.cuda_modules[filename] = drv.module_from_file(filename)

load_module.cuda_modules = dict()

def get_module(filename):
    """
    Retrive a loaded cuda module
    """
    return load_module.cuda_modules[filename]

def append_cuda_header(filename, define_dict):
    """
    Append the header to the CUDA kernel
    """
    with open(filename, 'r') as srcfile:
        src = srcfile.read()
    
    # Create macro header
    header = ""
    if define_dict is not None:
        for key in define_dict:
            header += "#define {} {}\n".format(key, define_dict[key])
    new_src = header + src
    return new_src

def load_module_from_code_str(cuda_code):
    cubin_filename = core.__JIT_CACHE_PATH__ + gen_cubin_filename(cuda_code)
    if os.path.exists(cubin_filename):
        debug.log(debug.INFO, TAG, "Find the cached cubin: " + cubin_filename)
    else:
        # Compile CUDA code to cubin
        with open(cubin_filename, "wb") as out_cubin:
            include_path = core.__OPENHD_MODULE_PATH__ + "/backend/include"
            cubin_data = pycuda.compiler.compile(
                    cuda_code, #nvcc="nvcc", cache_dir=None,
                    no_extern_c = True,
                    include_dirs=[include_path])
            out_cubin.write(cubin_data)

    load_module(cubin_filename)
    return cubin_filename


def gen_cubin_filename(cuda_code):
    return 'jit_' + str(hashlib.md5(str(cuda_code).encode('utf-8')).hexdigest()) + '.cubin'


def declare_2d_cuda_array(a): # a : 2d numpy array 
    assert(len(a.strides) == 2)

    # This error happens when we cannot access different features
    # with a single step of the data type size
    debug.thrifty_assert(
            a.dtype.itemsize == a.strides[1],
            "Error: data is not aligned. Try to align")

    a_N = np.int32(a.shape[0])
    a_stride = np.int32(a.strides[0])
    a_bytes = a.size * a.dtype.itemsize
    a_gpu = drv.mem_alloc(a_bytes)
    drv.memcpy_htod(a_gpu, a)
    return a_gpu, a_stride

def declare_cuda_array(a): # a : numpy array 
    debug.thrifty_assert(
            len(a.shape) <= 2,
            "Numpy array dimsion has to be 1 or 2."
            )

    if len(a.shape) == 2:
        return declare_2d_cuda_array(a)
    else:
        a_bytes = a.size * a.dtype.itemsize
        a_gpu = drv.mem_alloc(a_bytes)
        drv.memcpy_htod(a_gpu, a)
        return a_gpu, np.int32(0)

# Run this function twice before/after running a kernel
# tag is used only at the second run
def cuda_timing(tag=None):
    if cuda_timing.start is None:
        cuda_timing.start = drv.Event()
        cuda_timing.start.record() 
        return

    end = drv.Event()
    end.record()
    end.synchronize()
    sec = cuda_timing.start.time_till(end)*1e-3 # second

    end = None
    cuda_timing.start = None

    if tag is None:
        print("CUDA: %.5f" % (sec))
    else:
        print("CUDA: %.5f @ %s" % (sec, tag))

cuda_timing.start = None

def get_device_attrs(devicenum):
    device = drv.Device(devicenum)
    attrs = device.get_attributes()
    return attrs

def get_constant_memory_size(devicenum=0):
    attrs = get_device_attrs(devicenum)
    return attrs[drv.device_attribute.TOTAL_CONSTANT_MEMORY]

def page_lock_array(X):
    new_X = drv.pagelocked_empty(shape=X.shape, dtype=np.float32)
    new_X[:] = X
    return new_X


class Streamer(object):
    """ Create a stream for the 2d numpy arrary """
    def __init__(self, array, const_mem_addr, n_streamed_samples):
        # Create page-lock memory: TODO: do it offline
        self.array = page_lock_array(array)

        # transferred args
        self.const_mem_addr = const_mem_addr
        self.n_streamed_samples = n_streamed_samples # == __M__
        self.sample_size = array.shape[0]
        self.feature_size = array.shape[1]

        # Create stream
        self.N_STREAM = core.__N_STREAM__

        self.streams = []
        for _ in range(self.N_STREAM): 
            self.streams.append(drv.Stream())

        # Init loop vars
        self.iter_idx = 0
        self.strided_sample_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        served_stream_idx = self.iter_idx % self.N_STREAM
        self.iter_idx += 1

        if served_stream_idx == 0:
            if self.strided_sample_idx >= self.sample_size:
                drv.Context.synchronize()
                raise StopIteration()


            for s in range(self.N_STREAM):
                sidx = self.strided_sample_idx + self.n_streamed_samples * s
                if sidx >= self.sample_size:
                    continue

                drv.memcpy_htod_async(
                        self.const_mem_addr + \
                                s * self.n_streamed_samples * \
                                self.feature_size * \
                                np.dtype(self.array.dtype).itemsize,
                        self.array[sidx:sidx+self.n_streamed_samples, :],
                        self.streams[s])

            self.strided_sample_idx += self.N_STREAM * self.n_streamed_samples

        sample_idx = self.strided_sample_idx - \
                (self.N_STREAM - served_stream_idx) * self.n_streamed_samples

        # This is the same to the "sdix, s, stream[s]" above
        return sample_idx, served_stream_idx, self.streams[served_stream_idx]
