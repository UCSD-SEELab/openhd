"""
=====================
Test code 4: 
=====================
"""

import openhd as hd

D = 10000
hd.init(D=D)
@hd.run
def test4(F):
    id_base_hx = hd.hypermatrix(F)
    id_test = hd.hypervector()
    for f in range(F):
        id_base_hx[f] += id_test
        id_base_hx[f] += id_test

    return id_base_hx, id_test

@hd.run
def test4_2(F):
    id_base_hx = hd.hypermatrix(F)
    id_test = hd.draw_random_hypervector()
    for f in range(F):
        # In-place update
        id_base_hx[f] = id_test * hd.draw_random_hypervector() 

    for f in range(F):
        id_base_hx[f] += id_test * hd.draw_random_hypervector()
        id_base_hx[f] += id_test
        # Here to write

    t = 0
    # Here to read
    while True:
        if t < F:
            id_base_hx[f] += id_test
            id_base_hx[t] += id_base_hx[t]
            f = 1
            #continue
        else:
            break
        id_base_hx[t] += id_test

        t += 1

    return id_base_hx, id_test

@hd.run
def test4_3(F):
    id_base_hx = hd.hypermatrix(F)
    id_test = hd.hypervector()
    for f in range(F):
        id_test += id_base_hx[f]

    return id_base_hx, id_test

with hd.utils.timing("Fused IO test"):
    #id_base = test4(10)
    #id_base = test4_2(10)
    id_base = test4_3(10)
