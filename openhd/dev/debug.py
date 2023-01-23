"""
===================
Debugging utilities
===================

Debugging utilities for development
"""
from __future__ import print_function

import sys
from ..dev.bash_color import *

DEV = -1
INFO = 0
WARNING = 1
ERROR = 2

level_to_str = {
        DEV : 'DEV',
        INFO : 'INFO',
        WARNING : 'WARNING',
        ERROR : 'ERROR',
        }

level_to_color_ftn = {
        DEV : grn,
        INFO : blu,
        WARNING : wrn,
        ERROR : fal,
        }


def log(
        level, tag, log_str,
        add_new_line=True, add_lineno=False, print_once=False):
    """
    Print a debug log

    Args:
        level (int): DEV, INFO, WARNING, ERROR
    """

    if level >= log.cur_log_level:
        if add_new_line:
            log_str += '\n'

        if add_lineno:
            log_str = log_str.strip()
            elems = log_str.split('\n')
            new_log_str = "\n"
            for idx, line in enumerate(elems):
                new_log_str += "%3d  %s\n" % (idx+1, line)
            log_str = new_log_str


        log_message = "[%s]\t%s\t%s" % (level_to_str[level], tag, log_str)

        if print_once:
            if log_message in log.print_once_msgs:
                return
            log.print_once_msgs.add(log_message)

        if log.use_highlight:
            log_message = level_to_color_ftn[level](log_message)

        print(log_message, end="")

log.cur_log_level = WARNING 
log.use_highlight = True 
log.print_once_msgs = set()

def set_log_level(level, use_highlight=True):
    """
    Set the current logging level

    Args:
        level (int): INFO, WARNING, ERROR
    """
    log.cur_log_level = level
    log.use_highlight = True


class Logger(object):
    """ Simple logging class which supports a verbose mode """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.buf = ""

    def log(self, log_str, add_new_line=True):
        if add_new_line:
            log_str += '\n'

        if self.verbose:
            print(log_str, end="")

        self.buf += log_str

    def clear(self):
        self.buf = ""

    def get_log(self):
        return self.buf

def thrifty_assert(cond, message):
    """
    Check the condition and 
    if False, terminate with the meesage without stack logs
    """

    if not cond:
        log(ERROR, "OpenHD", message)
        sys.exit()
        assert(False) # Not reachable

