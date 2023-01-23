"""
================
Timing utilities
================
"""
import time
from ..backend.cuda_impl import drv

class timing:
    """
    Measure the execution time of a code section 

    Example:
        >>> with timing("Test"):
                ...
    """
    def __init__(self, TAG):
        self.TAG = str(TAG)

    def __enter__(self):
        self.ts = time.time()

    def __exit__(self, type, value, traceback):
        print(str(self.TAG) + "\t" + str(time.time() - self.ts))



def timing_function(func):
    """
    Decorator to measure the execution time of a function 

    Example:
        >>> @timing
            def foo(...):
    """

    def wrapper(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        print(str(func) + "- " + str(time.time() - ts))
        return result

    return wrapper

class timing_cuda_kernel:
    """
    Measure the execution time of a CUDA kernel

    Example:
        >>> with timing_cuda_kernel("Kernel name"):
                kernel_launch_code ...
    """
    def __init__(self, TAG):
        self.TAG = str(TAG)

    def __enter__(self):
        self.ts = time.time()
        self.start = drv.Event()
        self.start.record() 

    def __exit__(self, type, value, traceback):
        end = drv.Event()
        end.record()
        end.synchronize()
        sec = self.start.time_till(end)*1e-3 # second

        print(str(self.TAG) + "\t" + ("%.5f" % (time.time() - self.ts)) + \
                "\t" + ("%.5f" % sec))
