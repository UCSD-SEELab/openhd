"""
=====================
Test code 3: 
=====================
"""

import openhd as hd

D = 10000
hd.init(D=D)
@hd.run
def test3(F):
    id_base_hx = hd.hypermatrix(F)
    id_test = hd.draw_random_hypervector()
    for f in range(F):
        id_base_hx[f] = hd.draw_random_hypervector()
        id_test = id_base_hx[f]

    return id_base_hx, id_test

with hd.utils.timing("subscript test"):
    id_base = test3(10)
