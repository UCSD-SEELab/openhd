"""
===========
AST Scanner
===========

This performs the static analysis of the source ast tree to build
the annotation tree and the data flow graph.
"""
from .ast_annotator import annotate_ast
from .dfg_builder import build_data_flow_graph

TAG = "jit.ast_scanner"

def scan_ast(ast_root, arg_variables, encode_func_vars):
    """
    For the given ast of python code, create annotations & data flow graph
    """
    annotation_root = annotate_ast(ast_root, arg_variables, encode_func_vars)
    dfg_body, aNode_to_dNode = build_data_flow_graph(
            annotation_root, arg_variables)

    scanned_parcel = ScannedParcel(
            annotation_root,
            dfg_body, aNode_to_dNode
            )

    return scanned_parcel

class ScannedParcel(object):
    """
    The data sturucture delivered to the main jit compiler function
    """
    def __init__(self, annotation_root, dfg_body, aNode_to_dNode):
        self.aRoot = annotation_root
        self.dBody = dfg_body
        self.aNode_to_dNode = aNode_to_dNode

