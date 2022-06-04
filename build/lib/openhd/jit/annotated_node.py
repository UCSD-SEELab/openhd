"""
===============
Annotation Node
===============

This has the AnnotationNode built by ASTAnnotator as a tree structure
"""

import ast

from ..dev import debug
from ..jit import HYPERVECTOR_TYPE, HYPERMATRIX_TYPE
from ..jit import HD_INT_ARRAY_TYPE, HD_FLOAT_ARRAY_TYPE

TAG = "jit.annotated_node"

class AnnotationNode(object):
    """
    Analyzed annotation corresponding to a node of the ast tree
    """
    def __init__(self, item):
        self.item = item # AST node or field name
        self.child = None # list of AnnotationNode or AnnotationNode
        self.parent = None # AnnotationNode

        # Annotations
        self.transformable = False
        self.openhd_included = False

        self.item_type = None
        self.alloc_hv = False

    def set(self, child, transformable=None):
        if transformable is None:
            assert(child is not None)
            self.transformable = AnnotationNode.is_transformable(child)
        else:
            if child is not None:
                auto_transformable = AnnotationNode.is_transformable(child)
                self.transformable = transformable and auto_transformable
            else:
                assert(transformable is not None)
                self.transformable = transformable

        self.child = child

        # Update the parents
        if isinstance(self.child, list):
            for c in self.child:
                c.parent = self
        elif self.child is not None:
            self.child.parent = self

        return self

    @staticmethod
    def is_transformable(child):
        if isinstance(child, list):
            return all(AnnotationNode.is_transformable(subchild) \
                            for subchild in child)

        assert(isinstance(child, AnnotationNode))
        if not child.transformable:
            return False

        if child.child is None:
            return True

        return AnnotationNode.is_transformable(child.child)

    def set_tran(self, transformable):
        self.transformable = transformable
        return self

    def set_type(self, item_type):
        self.item_type = item_type

        def is_supported_type(item_type):
            return item_type in [
                    int, float, bool, tuple, 
                    HYPERVECTOR_TYPE, HYPERMATRIX_TYPE,
                    HD_INT_ARRAY_TYPE, HD_FLOAT_ARRAY_TYPE]

        if not is_supported_type(self.item_type):
            self.transformable = False

        return self

    def print_debug(self):
        msg = AnnotationNode._debug_str_gen(self, 0)
        debug.log(debug.DEV, TAG, '\n' + msg)

    @staticmethod
    def _debug_str_gen(aNode, indent): # Recursive call
        if aNode is None:
            return ""

        line = "  " * indent


        name = str(hex(id(aNode))) + " | "
        if isinstance(aNode.item, str):
            name += aNode.item
        else:
            name += str(type(aNode.item))

        var_name = str(getattr(aNode.item, 'id', ''))
        if len(var_name) > 0:
            name += " [%s]" % var_name
        line += name + " : "

        line += str(aNode.transformable)
        line += " + " + str(aNode.openhd_included)

        line += " < " + str(aNode.alloc_hv) + " > "

        if aNode.item_type:
            line += " == " + str(aNode.item_type)
        

        line += '\n'

        if isinstance(aNode.child, list):
            child_lines = ''.join(
                    AnnotationNode._debug_str_gen(c, indent + 1)
                    for c in aNode.child)
        else:
            child_lines = AnnotationNode._debug_str_gen(
                    aNode.child, indent + 1)
        return line + child_lines


    # Declare function
    hypervector_declare_functions = [
            # OpenHD declare functions
            'draw_random_hypervector',
            'draw_gaussian_hypervector',
            'hypervector',
            #'hypermatrix', # cannot be run on CUDA
            ]

    def is_included_declare(self):
        if isinstance(self.item, ast.Call):
            if isinstance(self.item.func, ast.Attribute):
                fname = self.item.func.attr # TODO: package name 
            elif isinstance(self.item.func, ast.Name):
                fname = self.item.func.id

            if fname in AnnotationNode.hypervector_declare_functions:
                return True

        if self.child is None:
            return False

        if isinstance(self.child, list):
            for c in self.child:
                if c.is_included_declare():
                    return True
            return False

        return self.child.is_included_declare()

    def flatten(self):
        """ Return the list of all annotation nodes included """
        ret = [self]

        if self.child is None:
            return ret

        if not isinstance(self.child, list):
            return ret + [self.child.flatten()]

        for c in self.child:
            ret += c.flatten()

        return ret

        




