"""
================
Permute Unnester
================

This transformates an ast tree (chunk) to unnest permute
"""
import ast
from copy import deepcopy
from ast import iter_fields, copy_location

from ..dev import debug

from ..jit import HYPERVECTOR_TYPE, HYPERMATRIX_TYPE

TAG = "jit.permute_unnester"

def unnest_permute(node, used_var_types):
    """ Unnest permute fuction """
    return PermuteUnnester(used_var_types).visit(node, None)
    
class PermuteUnnester(object):
    """
    Transform the cuda chunk function to unnest all permute functions
    """
    def __init__(self, used_var_types):
        self.used_var_types = used_var_types

    def visit(self, node, permutation_num_ast):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node, permutation_num_ast)

        assert(not(isinstance(ret, list) and len(ret) == 0))
        return ret

    def generic_visit(self, node, permutation_num_ast):
        """
        Called if no explicit visitor function exists for a node.
        """
        for field, old_value in iter_fields(node):
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value, permutation_num_ast)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value, permutation_num_ast)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


    def visit_Call(self, node, permutation_num_ast):
        fname = None
        if isinstance(node.func, ast.Attribute):
            fname = node.func.attr # TODO: package name - node.func.value.id
        elif isinstance(node.func, ast.Name):
            fname = node.func.id

        if fname != "permute":
            return self.generic_visit(node, permutation_num_ast)

        # If it's a permute function
        if node.args is None or len(node.args) != 2:
            debug.log(debug.ERROR, TAG, "openhd.permute requires two args.")
            assert(False)

        curr_perm_num_node = self.visit(node.args[1], permutation_num_ast)

        if permutation_num_ast is None:
            permutation_num_ast = curr_perm_num_node 
        else:
            permutation_num_ast = ast.BinOp(
                    op=ast.Add(),
                    left=permutation_num_ast,
                    right=curr_perm_num_node,
                    ctx=curr_perm_num_node.ctx)
        #print(curr_perm_num_node, permutation_num_ast)

        # Return the first argument, i.e., a hypervector (through arithmetics)
        return self.visit(node.args[0], permutation_num_ast)


    def visit_Name(self, node, permutation_num_ast):
        if permutation_num_ast is None:
            return self.generic_visit(node, permutation_num_ast)
        if not (self.used_var_types[node.id] == HYPERVECTOR_TYPE or \
                self.used_var_types[node.id] == HYPERMATRIX_TYPE):
            return self.generic_visit(node, permutation_num_ast)

        func = ast.Name("__permute__", ast.Load())
        args = [
                deepcopy(self.generic_visit(node, None)),
                deepcopy(permutation_num_ast)
                ]
        call = ast.Call(func, args, [])
        # call = ast.Call(func, args, [], None, None)
        
        return copy_location(call, node)
