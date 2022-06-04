"""
==================
Bash color utility
==================

Format texts in bash shell
"""

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def color_base(text, bcolor):
    return bcolor + text + ENDC

def blu(text):
    return color_base(text, OKBLUE)

def grn(text):
    return color_base(text, OKGREEN)


def bld(text):
    return color_base(text, BOLD)

def hdr(text):
    return color_base(text, HEADER)

def fal(text):
    return color_base(text, FAIL)

def wrn(text):
    return color_base(text, WARNING)

