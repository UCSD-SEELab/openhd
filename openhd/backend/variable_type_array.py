"""
===================
Variable Type Array
===================

This handles the arguments of the JIT kernel
"""
import numpy as np

import pycuda.gpuarray as gpuarray
from struct import pack, unpack

from ..dev import debug

from . import cuda_impl as ci

TAG = "backend.variable_type_array"

class VariableTypeArray(object):
    def __init__(self, size_in_bytes):
        self.size_in_bytes = size_in_bytes
        self.gpu_buffer = ci.drv.mem_alloc(size_in_bytes)
        self.cpu_buffer = bytearray(size_in_bytes)
        self.item_list = []

    def reset(self):
        self.item_list = []

    str_to_type = {
            'int' : int,
            'float' : float,
            'bool' : bool,
            }

    itemsize = {
            int : 4,
            float : 4,
            bool : 1,
                }

    type_to_struct_keyword = {
            int: 'i',
            float: 'f',
            bool: 'c',
            }


    def add(self, value, item_type):
        if len(self.item_list) == 0:
            offset = 0
        else:
            offset = self.itemsize[self.item_list[-1][0]] + \
                    self.item_list[-1][1]

        if isinstance(item_type, str):
            item_type = self.str_to_type[item_type]

            debug.thrifty_assert(
                    offset + self.itemsize[item_type] < self.size_in_bytes,
                    "Not sufficient argument array." + \
                            "Increase hd_arg_buffer_size in init()")

        self.item_list.append((item_type, offset, value))

    def addr(self, index):
        """
        Return address of the array element as GPUArray
        which is sendable to cuda kernal 
        """
        _, offset, __ = self.item_list[index]

        return gpuarray.GPUArray(
                (1,1),
                dtype=np.float, # this type is not important
                gpudata=(int(self.gpu_buffer) + offset))

    def push(self):
        # Copy Host to Device 
        for item_type, offset, value in self.item_list:
            self.cpu_buffer[offset:offset + self.itemsize[item_type]] = \
                    pack(self.type_to_struct_keyword[item_type], 
                            value)


        ci.drv.memcpy_htod(self.gpu_buffer, self.cpu_buffer)

    def pull(self):
        # Copy Device to Host 
        ci.drv.memcpy_dtoh(self.cpu_buffer, self.gpu_buffer)

    def __getitem__(self, index):
        item_type, offset, _ = self.item_list[index]

        return unpack(
                self.type_to_struct_keyword[item_type], 
                self.cpu_buffer[offset:offset + self.itemsize[item_type]])

