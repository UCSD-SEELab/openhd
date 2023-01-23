"""
============
JIT Compiler
============

This runs the jit compiler from a given python code
"""
import sys
import ast

from ..dev import debug
from .. import core

from ..jit import jit_codegen as codegen

from .ast_scanner import scan_ast
from .transpile_planner import plan_transpile
from .py_transformer import replace_python_code
from .cuda_code_generator import generate_pycuda_code

TAG = "jit.jit_compiler"

def compile(str_python_code, arg_variables, encode_func_vars=None):
    """
    Main function of this module:
    Compile a python function to the code running on CUDA

    Args:
    arg_variables = {'F': (int, 10), 'id_base': (HYPERMATRIX_TYPE, object), ...}
    """
    # 0. Keep the function arguments
    func_args = set(arg_variables.keys())

    # 1. Build the ast 
    node = ast.parse(str_python_code)
    node = ast.fix_missing_locations(node)
    node = unshell(node)
    #print(ast.dump(node))

    # 2. Scan ast
    scanned_parcel = scan_ast(node, arg_variables, encode_func_vars)

    # 3. Plan transpile
    chunk_list = plan_transpile(scanned_parcel, arg_variables, encode_func_vars)
    
    #print "<Before>"
    #print codegen.to_source(node)

    # 4. Generate cuda and python transformed code with cuda code compile
    py_code = replace_python_code(node, chunk_list, encode_func_vars)
    cuda_launcher_code, cubin_list = generate_pycuda_code(
            chunk_list, arg_variables, encode_func_vars)

    # 5. Check if the arg_variables are updated by the global variable 
    global_var_code = "\n"
    func_args = set(arg_variables.keys()) - func_args
    for var in func_args:
        val = core.__CONTEXT__[var]
        global_var_code += "%s = %s\n" % (var, val)

    return cuda_launcher_code + py_code + global_var_code, cubin_list

def unshell(node):
    """
    Unshell the decorator 
    """
    node.body[0].decorator_list = []
    return node

