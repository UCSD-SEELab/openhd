"""
=====================
Test code scratch
=====================
"""

import openhd as hd

# 1. Testing parameters ########################################################
D = 10000
hd.init(D=D)

# 2. Create base hyprevectors ##################################################
@hd.run
def test():
    #Q = (5,6,7)
    A, B, F = 1, 2, 3
    #L = [1]
    #
    #
    #F = 10.0
    #F *= 10
    #F = A / B
    #E = 1.1
    #F = E

    #Q = 1
    #X = True
    #T = True and Q and False

    #A[10] = 1
    #A = B[10:12]

    #F = 1
    #id_hvs = "str"

    #F = 10
    #Q = 2
    #L = [1,2,3]
    #id_hvs = hd.hv_matrix(F) # np.zeros(F, N) (not the empty list) 
    #for f in range(F/Q):
    #    id_hvs[f] = hd.permute(id_base, f)
    #    #L[0] = 2

    #with foo():
    #F = 10
    #F *= 1
    #A[10] = 1

    f = 0
    #f = f + 1 + 2
    D = 100
    #id_hvs = hd.hv_matrix(F) # np.zeros(F, N) (not the empty list) 
    id_hvs = hd.hypervector(F) # np.zeros(F, N) (not the empty list) 
    while f < F:
        if f < D:
            id_hvs = hd.permute(id_hvs, f)
            #id_hvs[f] = hd.permute(id_base, f)
            #id_hvs[f] = hd.permute(id_base, f)
        f += 1


    A = 'str'

    e = f
    for f in range(F):
        id_hvs = hd.permute(id_hvs, f)

test()
################################################################################
