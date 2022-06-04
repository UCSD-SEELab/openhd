"""
=====================
Test code 2: 
=====================
"""

import openhd as hd

D = 10000
hd.init(D=D)
@hd.run
def test2():
    f = 10
    e = 20
    #Q = 10 * (f + e)
    id_base = hd.draw_random_hypervector()
    level_base = hd.draw_random_hypervector()
    #id_base = hd.permute(hd.permute(id_base + level_base, f), 10 + e) * hd.permute(level_base, e+1)
    id_base = hd.permute(hd.permute(id_base + id_base, f), 10 + e) * hd.permute(level_base, e+1)
    return id_base

with hd.utils.timing("permute test"):
    id_base = test2()
