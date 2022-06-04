"""
=====================
Test code 1: 
=====================
"""

import openhd as hd

# 1. Testing parameters ########################################################
D = 10000
hd.init(D=D)

# 2. Create base hyprevectors ##################################################
@hd.run
def create_random_bases():
    id_base = hd.draw_random_hypervector()
    level_base = hd.draw_random_hypervector()
    return id_base, level_base

with hd.utils.timing("Create random bases"):
    id_base, level_base = create_random_bases()
    id_base.debug_print_values()
    level_base.debug_print_values()
################################################################################
