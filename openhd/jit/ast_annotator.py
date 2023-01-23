"""
===============
Annotation Node
===============

This module annotates a AST tree to identify JIT-compatible trees 
"""
import ast
from ast import iter_fields
from .annotated_node import AnnotationNode

from ..dev import debug

from .. import core
from ..core import hypervector

from ..jit import HYPERVECTOR_TYPE, HYPERMATRIX_TYPE
from ..jit import HD_INT_ARRAY_TYPE, HD_FLOAT_ARRAY_TYPE
from ..jit import NP_INT_ARRAY_TYPE, NP_FLOAT_ARRAY_TYPE
from ..jit import PREPROC_FEATURES

TAG = "jit.ast_annotator"

def annotate_ast(ast_root, arg_variables, encode_func_vars):
    """ Run ASTAnnotator """
    # Create virtual nodes to give information for the variables
    vnodes = VirtualKnownNode.factory_virtual_nodes(
            variables=arg_variables)

    # Run the annotator 
    visitor = ASTAnnotator(vnodes, arg_variables)
    annotation_root = visitor.visit(ast_root)

    # Traverse the tree to update the openhd_included
    update_openhd_included(annotation_root)

    annotation_root.print_debug()
    #assert(False)

    if encode_func_vars is not None:
        body_node = annotation_root.child[0].child[0].child[2]
        assert(body_node.item == "body")
        debug.thrifty_assert(
                body_node.transformable,
                "Encoding function should be run on CUDA " +
                "for full acceleration. Simplify your encode function.")

    return annotation_root

def is_hyper_type(item_type):
    return item_type in [HYPERVECTOR_TYPE, HYPERMATRIX_TYPE]


def update_openhd_included(aNode):
    aNode.openhd_included = is_hyper_type(aNode.item_type)

    if isinstance(aNode.child, list):
        for subchild in aNode.child:
            aNode.openhd_included |= update_openhd_included(subchild)
    elif aNode.child:
        aNode.openhd_included |= update_openhd_included(aNode.child)
   
    return aNode.openhd_included

class VirtualKnownNode(object):
    """
    Virtual node to handle each argument variable of the compiling function 
    """
    def __init__(self, name, item_type):
        self.id = name
        self.item_type = item_type
        self.lineno = -1

    @staticmethod
    def factory_virtual_nodes(variables):
        ret = []
        for n, (t, val) in variables.items():
            vNode = VirtualKnownNode(n, t)
            ret.append(vNode)

        return ret

class ASTAnnotator(object):
    """
    Performs a static analysis to build the AnnotationNode tree
    """
    boolean_constants = frozenset(['True', 'False'])
    known_functions = {
            # Function : (type, jit_availability)

            # Built-in python function 
            'range' : (int, True),
            'int' : (int, True),
            'float' : (float, True),
            'bool' : (bool, True),

            # OpenHD to declare
            'draw_random_hypervector': (HYPERVECTOR_TYPE, True),
            'draw_gaussian_hypervector': (HYPERVECTOR_TYPE, True),
            'hypervector' : (HYPERVECTOR_TYPE, True),
            'hypermatrix' : (HYPERMATRIX_TYPE, False), # cannot be run on CUDA

            # OpenHD built-in operators 
            'flip' : (HYPERVECTOR_TYPE, True),
            'permute' : (HYPERVECTOR_TYPE, True),
            'shuffle' : (HYPERVECTOR_TYPE, True), 
            
            # OpenHD built-in operators, but already compiled with AOT
            'search' : (HD_INT_ARRAY_TYPE, False), 
            'cossim' : (HD_FLOAT_ARRAY_TYPE, False), 
            'matmul' : (HYPERMATRIX_TYPE, False), 

            # OpenHD built-in operators which can be both AOT or JIT
            # Note: currently we only supports AOT and hypermatrix type
            'cos' : (HYPERMATRIX_TYPE, False),
            'sign' : (HYPERMATRIX_TYPE, False),


            # Numpy function to array - TODO: better inference for int type
            'array': (NP_FLOAT_ARRAY_TYPE, False),
            'zeros': (NP_FLOAT_ARRAY_TYPE, False),
            'ones': (NP_FLOAT_ARRAY_TYPE, False),

            # Cuda natives
            '__syncthreads': (int, True),
            '__hd_return': (int, True),
            }

    def __init__(self, known_nodes, arg_variables):
        self.declared_types = dict() # variable -> (declared_node, type)

        for vnode in known_nodes:
            self.declared_types[vnode.id] = \
                    (vnode, vnode.item_type)

        self.arg_variables = arg_variables

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)

        assert(not(isinstance(ret, list) and len(ret) == 0))
        return ret

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        aNode = AnnotationNode(node)
        if len(node._fields) == 0:
            return aNode

        children = []
        for field, value in iter_fields(node):
            aNode_field = AnnotationNode(field)
            if isinstance(value, list):
                if len(value) == 0:
                    children.append(aNode_field)
                    continue

                sub_children = []
                for item in value:
                    if isinstance(item, ast.AST):
                        aNode_field_list_item = self.visit(item)
                    else:
                        aNode_field_list_item = AnnotationNode(item)

                    sub_children.append(aNode_field_list_item)

                aNode_field.set(sub_children)
            elif isinstance(value, ast.AST):
                aNode_field.set(self.visit(value))

            children.append(aNode_field)

        return aNode.set(children)

    def visit_Num(self, node):
        return AnnotationNode(node).set(None, True).set_type(type(node.n))

    def visit_Str(self, node):
        return AnnotationNode(node).set(None, False).set_type(type(node.s))

    def visit_List(self, node):
        return AnnotationNode(node).set(None, False).set_type(list)

    def visit_Tuple(self, node):
        aNode = AnnotationNode(node)
        return aNode.set([self.visit(e) for e in node.elts]).set_type(tuple)

    def visit_Assign(self, node):
        aNode = AnnotationNode(node)

        if len(node.targets) != 1 or not(
                isinstance(node.targets[0], ast.Name) or \
                        isinstance(node.targets[0], ast.Tuple)):

            aNode_target = self.visit(node.targets[0])
            aNode_value = self.visit(node.value)

            if isinstance(node.targets[0], ast.Subscript):
                if aNode_target.child[0].item_type == NP_FLOAT_ARRAY_TYPE or \
                        aNode_target.child[0].item_type == NP_INT_ARRAY_TYPE:
                    if aNode_target.child[0].item.id != PREPROC_FEATURES:
                        aNode_target.set_tran(False)

            return aNode.set([aNode_target, aNode_value]) 


        aNode_target = self.visit(node.targets[0])
        aNode_value = self.visit(node.value)

        # New hypervector assignment
        if aNode_target.item_type is None:
            if aNode_value.is_included_declare():
                aNode_target.alloc_hv = True
            elif aNode_value.item_type == HYPERVECTOR_TYPE:
                aNode_target.alloc_hv = True

        # Copy data from value
        aNode_target.set_tran(aNode_value.transformable)
        aNode_target.set_type(aNode_value.item_type)

        if isinstance(node.targets[0], ast.Tuple):
            debug.thrifty_assert(
                    isinstance(node.value, ast.Tuple),
                    "Assigning tuple with one value is not supported yet.") 
            debug.thrifty_assert(
                    len(node.targets[0].elts) == len(node.value.elts),
                    "Tuple assignment must take the same # of elements") 

            for subtarget, subvalue, subtarget_node in zip(
                    aNode_target.child, aNode_value.child,
                    node.targets[0].elts):

                assert(isinstance(subtarget.item, ast.Name))
                if subtarget.item_type is None and subvalue.is_included_declare():
                    subtarget.alloc_hv = True

                # Copy data from value
                subtarget.set_tran(subvalue.transformable)
                subtarget.set_type(subvalue.item_type)

                self.declared_types[subtarget.item.id] = (
                        subtarget.item, subtarget.item_type)

                if isinstance(subtarget_node, ast.Subscript):
                    if subtarget.child[0].item_type == NP_FLOAT_ARRAY_TYPE or \
                            subtarget.child[0].item_type == NP_INT_ARRAY_TYPE:
                        subtarget.set_tran(False)

        else:
            self.declared_types[node.targets[0].id] = (
                    node.targets[0],
                    aNode_target.item_type)

            
        return aNode.set([aNode_target, aNode_value])

    def visit_Name(self, node):
        aNode = AnnotationNode(node)

        if node.id in self.boolean_constants:
            aNode.set(None, True).set_type(bool)
        elif node.id in self.declared_types:
            declared_node, item_type = self.declared_types[node.id]
            assert(declared_node.lineno <= node.lineno)
            aNode.set(None, True).set_type(item_type)
        elif node.id in core.__CONTEXT__: # Try to find it in the context
            val = core.__CONTEXT__[node.id]

            # Update arg vars and declared_node
            item_type = type(val)
            if isinstance(val, hypervector.hypervector):
                item_type = HYPERVECTOR_TYPE
            elif isinstance(val, hypervector.hypermatrix):
                item_type = HYPERMATRIX_TYPE

            self.arg_variables[node.id] = (item_type, val) 
            vnode = VirtualKnownNode(node.id, item_type)
            self.declared_types[vnode.id] = \
                    (vnode, vnode.item_type)

            # Do the same thing 
            declared_node, item_type = self.declared_types[node.id]
            aNode.set(None, True).set_type(item_type)
           
        return aNode

    def visit_Subscript(self, node):
        aNode = AnnotationNode(node)
        if not isinstance(node.slice, ast.Index):
            return self.generic_visit(node)

        aNode_value = self.visit(node.value)
        aNode_slice = self.visit(node.slice)

        transformable = False
        item_type = None
        if aNode_value.item_type == HYPERMATRIX_TYPE:
            transformable = True
            item_type = HYPERVECTOR_TYPE
        elif aNode_value.item_type == NP_INT_ARRAY_TYPE:
            aNode_value.transformable = True # override
            transformable = True
            item_type = int
        elif aNode_value.item_type == NP_FLOAT_ARRAY_TYPE:
            aNode_value.transformable = True # override
            transformable = True
            item_type = float
        elif aNode_value.item_type == HD_INT_ARRAY_TYPE:
            aNode_value.transformable = True # override
            transformable = True
            item_type = int
        elif aNode_value.item_type == HD_FLOAT_ARRAY_TYPE:
            aNode_value.transformable = True # override
            transformable = True
            item_type = float


        aNode.set([aNode_value, aNode_slice], transformable)
        if item_type is not None:
            aNode.set_type(item_type)

        return aNode

    def visit_Index(self, node):
        aNode = AnnotationNode(node)
        return aNode.set([self.visit(node.value)])

    def visit_Constant(self, node):
        return AnnotationNode(node).set(None, True).set_type(type(node.n))

    @staticmethod
    def type_inference(item1, item2):
        # Return type
        def is_either_type(item1, item2, item_type):
            if item1.item_type == item_type:
                return True

            if item2.item_type == item_type:
                return True

            return False

        #type_order = [HYPERMATRIX_TYPE, HYPERVECTOR_TYPE, float, int, bool]
        if is_either_type(item1, item2, HYPERMATRIX_TYPE):
            if item1.item_type == item2.item_type:
                assert(item1.hx_size == item2.hx_size)
                return HYPERMATRIX_TYPE
            elif item1.item_type == HYPERMATRIX_TYPE:
                return HYPERMATRIX_TYPE
            else: # item2.item_type == HYPERMATRIX_TYPE
                return HYPERMATRIX_TYPE
        elif is_either_type(item1, item2, HYPERVECTOR_TYPE):
            return HYPERVECTOR_TYPE

        primitive_type_order = [float, int, bool]
        for item_type in primitive_type_order:
            if is_either_type(item1, item2, item_type):
                return item_type

        return None

    def visit_AugAssign(self, node):
        aNode = AnnotationNode(node)
        if isinstance(node.target, ast.Name):
            node_name = node.target.id
        elif isinstance(node.target, ast.Subscript):
            node_name = node.target.value.id
        else:
            assert(False)
            return self.generic_visit(node)

        if node_name not in self.declared_types: # in fact a syntax error
            return self.generic_visit(node)

        aNode_target = self.visit(node.target)
        aNode_value = self.visit(node.value)

        aNode_target.set_tran(aNode_value.transformable)

        # update the type of the target 
        if isinstance(node.target, ast.Name):
            item_type = ASTAnnotator.type_inference(
                    aNode_target, aNode_value)
            aNode_target.set_type(item_type)
            self.declared_types[node_name] = (
                    self.declared_types[node_name][0],
                    aNode_target.item_type)
        elif isinstance(node.target, ast.Subscript):
            self.declared_types[node_name] = (
                    self.declared_types[node_name][0],
                    aNode_target.child[0].item_type)
        else:
            assert(False)

        return aNode.set([aNode_target, aNode_value])


    def visit_Call(self, node):
        aNode = AnnotationNode(node)

        fname = None
        if isinstance(node.func, ast.Attribute):
            fname = node.func.attr # TODO: package name - node.func.value.id
        elif isinstance(node.func, ast.Name):
            fname = node.func.id

        if fname in self.known_functions:
            item_type, jit_availability = self.known_functions[fname]
            if node.args:
                children_args = [self.visit(a) for a in node.args]
                aNode.set(children_args, jit_availability).set_type(item_type)
            else:
                # TODO: Check the arguments
                # Currently, it just tries to compile it
                # without guaranteeing the syntax correctness
                aNode.set(None, jit_availability).set_type(item_type)

            return aNode
        
        # Skip the validation for the function name
        #return self.generic_visit(node)
        return aNode.set([self.generic_visit(a) for a in node.args])

    def visit_BinOp(self, node):
        aNode = AnnotationNode(node)
        child = [self.visit(node.left), self.visit(node.right)]
        item_type = ASTAnnotator.type_inference(child[0], child[1])

        return aNode.set(child).set_type(item_type)

    def visit_BoolOp(self, node):
        aNode = AnnotationNode(node)
        child = [self.visit(v) for v in node.values]
        return aNode.set(child).set_type(bool)

    def visit_Compare(self, node):
        aNode = AnnotationNode(node)
        assert(len(node.comparators) == 1)
        assert(len(node.ops) == 1)

        left = self.visit(node.left)
        right = self.visit(node.comparators[0])

        return aNode.set([left, right])


    def visit_For(self, node):
        aNode = AnnotationNode(node)

        if not isinstance(node.target, ast.Name):
            return self.generic_visit(node)

        if not isinstance(node.iter, ast.Call):
            return self.generic_visit(node)

        if node.iter.func.id != "range":
            return self.generic_visit(node)

        target = self.visit(node.target)
        it = self.visit(node.iter)
        assert(it.item_type == int)

        target.set_tran(it.transformable)
        target.set_type(it.item_type)
        self.declared_types[node.target.id] = (node.target, target.item_type)

        aNode_body = AnnotationNode(node.body)
        body = aNode_body.set([self.visit(c) for c in node.body])

        aNode.set([target, it, body])

        return aNode

    def visit_While(self, node):
        aNode = AnnotationNode(node)

        test = self.visit(node.test)
        aNode_body = AnnotationNode(node.body)
        body = aNode_body.set([self.visit(c) for c in node.body])

        aNode.set([test, body])
        return aNode

    def visit_Break(self, node):
        aNode = AnnotationNode(node)
        return aNode.set(None, True)

    def visit_Continue(self, node):
        aNode = AnnotationNode(node)
        return aNode.set(None, True)

    def visit_If(self, node):
        aNode = AnnotationNode(node)

        test = self.visit(node.test)
        body = AnnotationNode(node.body)
        body.set([self.visit(c) for c in node.body])

        if len(node.orelse) == 0:
            aNode.set([test, body])
        else:
            orelse = AnnotationNode(node.orelse)
            orelse.set([self.visit(c) for c in node.orelse])
            aNode.set([test, body, orelse])

        return aNode

