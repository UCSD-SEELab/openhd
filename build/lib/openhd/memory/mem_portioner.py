"""
================
Memory portioner
================

This provides classes to portion the memory resource (e.g., shared/constant mem)
for generating the kernel code
"""
import numpy as np

from ..dev import debug

from ..backend import cuda_impl as ci

TAG = "memory.mem_portioner"
    
class ConstantMemPortioner(object):
    def __init__(self):
        self.total_mem_size = ci.get_constant_memory_size(devicenum=0)
        self.used_mem_size = 0

    def get_available_mem_size(self):
        return self.total_mem_size - self.used_mem_size

    def allocate(self, name, size_str, vars_in_size, item_type):
        """
        Allocate a memory variable with the name and the size

        Example:
            const_portioner.allocate(
                FEATURE_STREAM,
                "__N_STREAM__ * __M__ * __F__",
                {
                    "__N_STREAM__": __N_STREAM__,
                    "__M__": __M__,
                    "__F__": __F__
                })
        """

        # Calculate the memory size
        eval_size_str = size_str
        for var, size in vars_in_size.items():
            eval_size_str = eval_size_str.replace(var, str(size))
        size = eval(eval_size_str)

        debug.thrifty_assert(
                size > 0,
                "The array allocated in constant memory must be larger than 0")

        debug.thrifty_assert(
                self.get_available_mem_size() >= size,
                "Not enough constant memory")

        # Item type
        itemsize = np.dtype(item_type).itemsize
        if item_type == np.float32: 
            item_type_str = "float"
        elif item_type == np.int32: 
            item_type_str = "int"
        elif item_type == np.uint32: 
            item_type_str = "unsigned int"
        elif item_type == np.ushort: 
            item_type_str = "unsigned short"
        elif item_type == np.bool: 
            item_type_str = "bool"
        else:
            raise NotImplementedError

        # Update the size
        self.used_mem_size += size * itemsize

        return "__device__ __constant__ %s %s[%s];\n" % (
                item_type_str, name, size_str)
