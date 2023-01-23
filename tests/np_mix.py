"""
==========================
Test code - numpy mix code
==========================
"""

import openhd as hd

D = 10000
hd.init(D=D)

@hd.run
def test():
    a = np.zeros((10,2))
    a[0,1] = 2

    f = 1
    level_base = hd.draw_random_hypervector()
    id_base = level_base * a[0,f]
    id_base += level_base * a[0,f]
    return id_base, a

with hd.utils.timing("np mix test"):
    id_base, a = test()
