"""
====================
Check PyCuda support 
====================

Importing this file checks if pycuda is installed in the system
"""


import imp
try:
    imp.find_module('pycuda')
    USE_CUDA = True
except ImportError:
    USE_CUDA = False

#USE_CUDA = False
