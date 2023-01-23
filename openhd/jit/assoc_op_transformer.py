"""
==========================
Associative OP Transformer
- NOT USED
==========================

This converts assignment functions related to associative operations 
so that the results are handled with pointers
"""
import ast

from ..dev import debug

TAG = "jit.assoc_op_transformer"

def convert_assoc_op(node):
    """ Convert shuffle calls """
    return AssocOpTransformer().visit(node)
    
class AssocOpTransformer(ast.NodeTransformer):
    """
    Convert function calls:
    C = hd.search(A, B)
    hd.search(A, B, C)
    """
    def visit_Assign(self, node):
        value_node = node.value

        if not isinstance(value_node, ast.Call):
            return self.generic_visit(node)

        if isinstance(value_node.func, ast.Attribute):
            fname = value_node.func.attr # TODO: package name - node.func.value.id
        elif isinstance(value_node.func, ast.Name):
            fname = value_node.func.id

        assoc_op_functions = ["search"]

        if fname not in assoc_op_functions:
            return self.generic_visit(node)

        node_C = node.targets[0]
        value_node.args.append(node_C)

        return node

