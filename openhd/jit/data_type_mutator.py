"""
==================
Data Type Mutation
==================

This performs inferences for the data type (element types) of hypervectors
to optimize the memory size of hypervectors
"""

import numpy as np
import sympy as sp


import ast
from ast import iter_fields
from collections import defaultdict

from ..dev import debug

from ..jit import jit_codegen as codegen

from ..jit import HYPERVECTOR_TYPE, HYPERMATRIX_TYPE, HYPERELEMNT_TYPE
from ..jit import ARG_PREFIX, STRIDE_POSTFIX
from ..jit import NP_INT_ARRAY_TYPE, NP_FLOAT_ARRAY_TYPE


TAG = "jit.date_type_mutator"

def mutate_data_type(node, chunk, arg_variables):
    """ Data type mutation """

    # Debugging code
    debug.log(debug.DEV, TAG, "<<--------------------------")
    debug.log(debug.DEV, TAG, str(ast.dump(node)))
    debug.log(
            debug.DEV, TAG,
            codegen.to_source(node), add_lineno=True)
    for var, var_type in chunk.used_var_types.items():
        debug.log(debug.DEV, TAG, ">> %s : %s" % (var, str(var_type)))
    debug.log(debug.DEV, TAG, "-------------------------->>")
    # End of debugging

    # 1. Convert node to expression tree
    expTreeGenerator = ExpressionTreeGenerator(chunk)
    expTree = expTreeGenerator.visit(node)
    expTree.print_debug()


    for var, var_type in chunk.used_var_types.items():
        #if ARG_PREFIX not in var:
        #    continue
        if var_type == HYPERMATRIX_TYPE or var_type == HYPERVECTOR_TYPE:
            if ARG_PREFIX + var in chunk.used_var_types:
                continue
            chunk.used_var_types[var] = "float*"

        elif var_type == HYPERELEMNT_TYPE:
            chunk.used_var_types[var] = float



    return node


def is_hyper_type(item_type):
    return item_type in [HYPERVECTOR_TYPE, HYPERMATRIX_TYPE, HYPERELEMNT_TYPE]

# TODO: Maybe move to anotehr file
class ExpTree(object):
    def __init__(self, node):
        # possible type for Node
        # 1. ast.Node, 2. string (if-statement) or dominator overriding)
        self.node = node 
        self.statement = None # (var_name::string, expr)
        self.node_to_repeat = None
        self.children = []

    def add(self, child):
        if child is None:
            return

        assert(isinstance(child, ExpTree))
        self.children.append(child)
        return self

    def set_statement(self, var, expr):
        self.statement = (var, expr)
        return self

    def print_debug(self):
        msg = ExpTree._debug_str_gen(self, 0)
        debug.log(debug.ERROR, TAG, '\n' + msg)

    @staticmethod
    def _debug_str_gen(eTree, indent): # Recursive call
        line = "  " * indent

        name = ""
        if isinstance(eTree.node, str):
            name += eTree.node
        else:
            name += str(type(eTree.node))

        line += name + " : "

        if eTree.statement is not None:
            var, expr = eTree.statement
            line += " [%s : %s]" % (var, str(expr))
            assert(len(eTree.children) == 0)

        if eTree.node_to_repeat is not None:
            if isinstance(eTree.node_to_repeat, ast.Name):
                arg_str = eTree.node_to_repeat.id
            else:
                arg_str = eTree.node_to_repeat.n

            line += " <RPT: %s>" % (arg_str)

        line += '\n'

        child_lines = ''.join(
                ExpTree._debug_str_gen(c, indent + 1)
                for c in eTree.children)

        return line + child_lines

class ExpressionTreeGenerator(object):
    def __init__(self, chunk):
        self.chunk = chunk
        self.symdict = dict()

        # Scala dominators
        self.__booldmt__ = sp.symbols("__booldmt__", integer=True)
        self.symdict["__booldmt__"] = self.__booldmt__
        self.__intdmt__ = sp.symbols("__intdmt__", integer=True)
        self.symdict["__intdmt__"] = self.__intdmt__
        self.__floatdmt__ = sp.symbols("__floatdmt__", integer=True)
        self.symdict["__floatdmt__"] = self.__floatdmt__

        self.__zero__ = sp.symbols("__zero__", integer=True)
        self.symdict["__zero__"] = self.__zero__
        self.__one__ = sp.symbols("__one__", integer=True)
        self.symdict["__one__"] = self.__one__


        # Function type mapping - {ftnname : overriding_symbol}
        # If the symbol is None, it will bypass the type from the argument
        self.ftn_to_symbol = {
            # Built-in python function 
            'int' : self.__intdmt__,
            'float' : self.__floatdmt__,
            'bool' : self.__booldmt__,

            # OpenHD to declare
            'draw_random_hypervector': self.__one__,
            'draw_gaussian_hypervector': self.__floatdmt__,
            'hypervector' : self.__zero__,

            # Functions not included in the chunk
            #'hypermatrix' : (HYPERMATRIX_TYPE, False)

            # Functions called only for assigned target
            #'*__hx_shuffle__' : None, 

            # OpenHD built-in operators - bypass the type
            'flip' : None,
            '*__permute__' : None, # The name was changed when unnesting
            '*__hvdim__': None,
            '*__hxdim__': None,

            # TODO: npdim 
            }

        # The followings must cover all ast nodes that may change the HV type
        # See also is_transformable_ast in "transpile_planner"
        self.evalutable_node = [
                ast.Assign,
                ast.AugAssign,
                ast.While,
                ast.If,
                ast.For,
                #ast.Call,
                #ast.Expr
                ]


    def get_sym(self, var_name):
        if var_name in self.symdict:
            return self.symdict[var_name]

        new_sym = sp.symbols(var_name)
        self.symdict[var_name] = new_sym
        return new_sym

    def visit(self, node):
        # node: ast node
        # return: inference function expression (sub)tree or
        # expression in a statement
        """ Visit an AST node """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def visit_body(self, node_list, eTree):
        for node in node_list:
            if not any(isinstance(node, t) for t in self.evalutable_node):
                continue
            subTree = self.visit(node)
            if subTree is None:
                continue
            eTree.add(subTree)

        return eTree

    def visit_Module(self, node):
        eTree = ExpTree(node)
        return self.visit_body(node.body[0].body, eTree)

    def visit_For(self, node):
        # Currently, it can be statically analyzed 
        # only when it is called with range in a predictable variable
        try:
            cond = isinstance(node.target, ast.Name)
            cond &= isinstance(node.iter, ast.Call)
            cond &= node.iter.func.id == "range"
            arg = node.iter.args[0]
            cond &= isinstance(arg, ast.Name) or (arg, ast.Num)
        except:
            # override all variables to inf like while loop
            return self.visit_While(node) 

        if not cond:
            return self.visit_While(node) 
            
        eTree = ExpTree(node)
        eTree.node_to_repeat = arg
        eTree = self.visit_body(node.body, eTree)
        if len(eTree.children) == 0:
            return None

        return eTree

    def visit_While(self, node):
        eTree = ExpTree(node)
        eTree = self.visit_body(node.body, eTree)

        # Dominator overriding:
        # all assigned variables in the tree will be __floatdmt__ and flattened
        all_vars = set()
        def collect_all_vars(eTree, all_vars):
            if eTree.statement is not None:
                all_vars.add(eTree.statement[0])

            for sub_eTree in eTree.children:
                collect_all_vars(sub_eTree, all_vars)

        collect_all_vars(eTree, all_vars)
        eTree_flatten = ExpTree(node)
        for var in all_vars:
            eTree_flatten.add(
                    ExpTree("OVERRIDING").set_statement(var, self.__floatdmt__))

        if len(eTree_flatten.children) == 0:
            return None

        return eTree_flatten 

    def visit_If(self, node):
        eTree = ExpTree(node)

        eTree_then = ExpTree('then')
        eTree_then = self.visit_body(node.body, eTree_then)
        eTree_else = ExpTree('else')
        eTree_else = self.visit_body(node.orelse, eTree_else)

        if len(eTree_then.children) == 0 and len(eTree_else.children) == 0:
            return None

        eTree.add(eTree_then)
        eTree.add(eTree_else)
        return eTree

    def visit_Assign(self, node):
        # This function returns only when assigning HYPERVECTOR or MATRIX
        if len(node.targets) != 1:
            # NOTE: When it happens?
            assert(False) 

        target_node = node.targets[0]

        if isinstance(target_node, ast.Call):
            # Case 1: Assignment to pointer retrieved by built-in access ftns
            ftn_name = target_node.func.id
            if ftn_name not in ["*__hxdim__", "*__hvdim__", "*__hx_shuffle__"]:
                return None # Skip evaluation for non-hv type assignment
            var_name = target_node.args[0].id
        elif isinstance(target_node, ast.Name) and node.targets[0].id:
            # Case 2: Assignment to hypervector element
            var_name = target_node.id
            if not self.chunk.used_var_types[var_name] == HYPERELEMNT_TYPE:
                return None
        else:
            return None

        expr = self.visit(node.value)
        if var_name == str(expr): # Assign the same
            return None

        return ExpTree(node).set_statement(var_name, expr)

    def visit_AugAssign(self, node):
        # Convert it to normal assignment to evalute
        return self.visit(
                ast.Assign(
                    [node.target],
                    ast.BinOp(
                        op=node.op,
                        left=node.target,
                        right=node.value
                        )
                    )
                )

    def visit_BinOp(self, node):
        left_expr = self.visit(node.left)
        rght_expr = self.visit(node.right)

        def is_either(left_expr, right_expr, symbol):
            return left_expr == symbol or right_expr == symbol

        # Early type domination
        # NOTE: Bool may not dominate the type
        if is_either(left_expr, rght_expr, self.__floatdmt__):
            return self.__floatdmt__
        if is_either(left_expr, rght_expr, self.__intdmt__):
            return self.__intdmt__

        if isinstance(node.op, ast.Mult):
            return left_expr * rght_expr
        elif isinstance(node.op, ast.Add):
            return left_expr + rght_expr
        elif isinstance(node.op, ast.Sub):
            # In the range computation, the subtraction can be regarded as +
            return left_expr + rght_expr 
        elif isinstance(node.op, ast.Div): 
            # concise computation is not supported yet 
            # Instead, use float dominator
            return self.__floatdmt__
        else:
            debug.log(debug.ERROR, TAG, "Unsupported operator: " % str(node.op))
            assert(False)

    def visit_Num(self, node):
        if isinstance(node.n, int):
            return self.__intdmt__
        elif isinstance(node.n, float):
            return self.__floatdmt__
        else:
            assert(False)

    boolean_constants = frozenset(['True', 'False'])
    def visit_Name(self, node):
        var_name = node.id
        if var_name in self.boolean_constants:
            return self.warn_bool_node(node)

        debug.thrifty_assert(
                var_name in self.chunk.used_var_types,
                "Cannot find the variable type of [%s]" % var_name
                )

        var_type = self.chunk.used_var_types[var_name]
        if var_type == int:
            return self.__intdmt__
        elif var_type == float:
            return self.__floatdmt__
        elif var_type == bool:
            return self.warn_bool_node(node)
        elif is_hyper_type(var_type):  
            # NOTE: NP_INT_ARRAY_TYPE or HD_FLOAT_ARRAY_TYPE, etc, must be
            # handled with Subscript
            return self.get_sym(var_name)
        else:
            print(var_name, var_type)
            assert(False)

    def visit_Subscript(self, node):
        var_name = node.value.id
        var_type = self.chunk.used_var_types[var_name]

        if var_type == NP_INT_ARRAY_TYPE or HD_INT_ARRAY_TYPE:
            return self.__intdmt__
        elif var_type == NP_FLOAT_ARRAY_TYPE or HD_FLOAT_ARRAY_TYPE:
            return self.__floatdmt__
        elif var_type == HYPERMATRIX_TYPE: 
            return self.get_sym(var_name)
        else:
            assert(False)

        return self.get_sym(node.id)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            fname = node.func.attr # TODO: package name - node.func.value.id
        elif isinstance(node.func, ast.Name):
            fname = node.func.id

        if fname not in self.ftn_to_symbol:
            debug.log(
                    debug.ERROR, TAG,
                    "Unrecognizable function in hypervector computation: %s" % \
                            fname)
            assert(False)

        symbol = self.ftn_to_symbol[fname]
        if symbol is None:
            symbol = self.visit(node.args[0])
            debug.thrifty_assert(
                    symbol in self.symdict.values(),
                    "The first argument of the function %s is not an HV type" \
                            % fname)


        return symbol


    def visit_BoolOp(self, node):
        return self.warn_bool_node(node)

    def warn_bool_node(self, node):
        MSG = "Ast node that returns boolean scala dominator is not "
        MSG += "fully tested yet."
        debug.log(debug.WARNING, TAG, MSG)
        return self.__booldmt__

    # The below ast types cannot be included in the jit-available chunk
    def visit_Expr(self, node):
        assert(False)

    def visit_Str(self, node):
        assert(False)

    def visit_List(self, node):
        assert(False)

    def visit_Tuple(self, node):
        assert(False)

    def visit_Compare(self, node):
        assert(False)

    def generic_visit(self, node):
        # must be unrechable
        debug.log(debug.ERROR, TAG, "Unrecognizable node: %s" % str(node))
        assert(False)

