"""
===========
OpenHD Core
===========
"""

# Module path: initialized in init
__OPENHD_MODULE_PATH__ = ""

__JIT_CACHE_PATH__ = ""

# Hypervector dimension
_D = None

# Context
__CONTEXT__ = None

# Stream
__N_STREAM__ = 2

# Shuffle indices (copied to the GPU memory if needed)
__SHUFFLE_INDICES_CPU__ = None

# Memory management
__ARRAY_POOL_MANAGER__ = None
