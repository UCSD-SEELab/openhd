"""
===================
HD name transformer
===================

This maps the OpenHD functions and variables in python to the Cuda variables 
"""

import ast

from ..dev import debug
from ..jit import cuda_builtin_vars

from ..jit import ARG_PREFIX, FEATURE_STREAM

TAG = "jit.hd_name_transformer"


def map_hd_names(node, encode_function_vars):
    """ Map HD functions """
    transformer = HDNameTransformer(encode_function_vars)
    new_node = transformer.visit(node)
    return new_node, transformer.used_func_names

class HDNameTransformer(ast.NodeTransformer):
    hd_op_func_map = {
            'hypervector' : '__hypervector__',
            'draw_random_hypervector' : '__draw_random_hypervector__',
            'draw_gaussian_hypervector' : '__draw_gaussian_hypervector__',
            'flip' : '__flip__',
            }

    other_func_map = {
            '__hd_return' : ast.Return(value=None)
            }

    def __init__(self, encode_function_vars):
        self.encode_function_vars = encode_function_vars
        self.used_func_names = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            fname = node.func.attr # TODO: package name - node.func.value.id
        elif isinstance(node.func, ast.Name):
            fname = node.func.id

        if fname[0] == '*': # remove the pointer access
            fname = fname[1:]
        self.used_func_names.add(fname)

        if fname in self.other_func_map:
            return self.other_func_map[fname]

        if fname not in self.hd_op_func_map:
            return self.generic_visit(node)

        # Change to the cuda function name
        node.func = ast.Name(self.hd_op_func_map[fname], ast.Load())
        self.used_func_names.add(self.hd_op_func_map[fname])

        # Add dimension variable
        if node.args is None:
            node.args = []

        node.args.append(ast.Name("__d__", ast.Load()))

        return node

    def visit_Name(self, node):
        if node.id in cuda_builtin_vars:
            node.id = cuda_builtin_vars[node.id]

        if self.encode_function_vars is not None:
            if node.id == ARG_PREFIX + self.encode_function_vars[0]:
                node.id = FEATURE_STREAM

        return node

