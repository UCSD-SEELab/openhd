"""
===================
Shuffle Transformer
===================

This converts `shuffled' assignment statements to the write call
"""
import ast

from ..dev import debug

TAG = "jit.shuffle_transformer"

def convert_shuffle(node):
    """ Convert shuffle calls """
    return ShuffleTransformer().visit(node)
    
class ShuffleTransformer(ast.NodeTransformer):
    """
    Convert shuffle calls:
    A = hd.shuffle(B)
    hd.shuffle(A) = B
    """
    def visit_Assign(self, node):
        value_node = node.value

        if not isinstance(value_node, ast.Call):
            return self.generic_visit(node)

        if isinstance(value_node.func, ast.Attribute):
            fname = value_node.func.attr # TODO: package name - node.func.value.id
        elif isinstance(value_node.func, ast.Name):
            fname = value_node.func.id

        if fname != "shuffle":
            return self.generic_visit(node)

        node_A = node.targets[0]
        node_B = value_node.args[0]

        value_node.args[0] = node_A

        node.targets = [value_node]
        node.value = node_B

        return node

