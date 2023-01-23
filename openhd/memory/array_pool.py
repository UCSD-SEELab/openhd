"""
=================
OpenHD Array Pool
=================

Memory pool management for OpenHD Array

seealso: core.array
"""

from ..dev import debug

from .. import core
from ..backend import cuda_impl as ci

from collections import OrderedDict 

TAG = "memory.array_pool"

class ArrayPoolManager(object):
    """
    Created in openhd.init()
    """
    def __init__(self, max_array_cnt):
        self.max_array_cnt = max_array_cnt
        self.pool_size = 0
        self.lru_addrs = OrderedDict() # addr -> (size, used)

    def alloc(self, elem_cnt):
        addr = None
        size = elem_cnt * 4
        for _addr, (_size, _used) in self.lru_addrs.items():
            if size == _size and _used == False:
                # Update LRU
                addr = _addr
                self.lru_addrs.pop(_addr)
                self.lru_addrs[_addr] = (_size, True) 

                #debug.log(
                #            debug.DEV, TAG,
                #            "Reuse an address (size: %d) from pool" % _size)
                break

        if addr is None: # if no same size exists
            # Create a new item
            addr = ci.drv.mem_alloc(size)
            self.lru_addrs[addr] = (size, True)
            self.pool_size += size

        self.collect_garbage()
        return addr

    def free(self, addr):
        assert(addr in self.lru_addrs)
        _size, _ = self.lru_addrs[addr]
        self.lru_addrs[addr] = (_size, False)

        self.collect_garbage()

    def collect_garbage(self):
        while self.max_array_cnt < len(self.lru_addrs):
            found = False
            for _addr, (_size, _used) in self.lru_addrs.items():
                if not _used:
                    self.lru_addrs.pop(_addr)
                    #debug.log(
                    #        debug.DEV, TAG,
                    #        "Remove an address (size: %d) from pool" % _size)
                    self.pool_size -= _size
                    found = True
                    break

            if found is False:
                return


def alloc(elem_cnt):
    return core.__ARRAY_POOL_MANAGER__.alloc(elem_cnt)

def free(addr):
    return core.__ARRAY_POOL_MANAGER__.free(addr) 
