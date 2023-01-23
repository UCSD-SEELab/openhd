"""
===========
CFG builder
===========

This builds the control flow graph
"""

import ast

from ..dev import debug

from .cfg_node import CFGNode

TAG = "jit.cfg_builder"

def build_control_flow_graph(ast_root):
    """ Build the control flow graph from an AST node """
    builder = CFGBuilder()
    body_nodes = unshell(ast_root)

    start_cNode = builder.build(body_nodes)
    #start_cNode.visualize_debug()
    return start_cNode


def unshell(ast_root):
    """ Return the body node list of the function """
    return ast_root.body[0].body

class CFGBuilder(object):
    def __init__(self):
        # Start and end node are not inserted to node_to_cNode 
        self.start_node = CFGNode("START")
        self.end_node = CFGNode("END")

        self.node_to_cNode = dict() 

    def create_cfg_node(self, node):
        cNode = CFGNode(node) 
        self.node_to_cNode[node] = cNode
        return cNode

    def build(self, body_nodes):
        # Build the CFG by connecting the body nodes 
        last_cNode = self.connect_node_list(
                self.start_node, body_nodes, None, None)

        if last_cNode is not None:
            self.end_node.add_incoming_edge(last_cNode)

        return self.start_node

    def connect_node_list(self, start_cNode, node_list, entry, exit):
        # build the sequential CFG
        prev_cNode = start_cNode
        for node in node_list: 
            cNode = self.visit(node, prev_cNode, entry, exit)
            if cNode is None:
                return None

            prev_cNode = cNode

        return prev_cNode # Last node

    def visit(self, node, prev_node, entry, exit):
        """ Visit an AST node ."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        cNode = visitor(node, prev_node, entry, exit)
        return cNode

    def generic_visit(self, node, prev_cNode, entry, exit):
        cNode = self.create_cfg_node(node)
        cNode.add_incoming_edge(prev_cNode)
        return cNode

    def visit_For(self, node, prev_cNode, entry, exit):
        entry_node = CFGNode("ENTRY")
        exit_node = CFGNode("EXIT")

        entry_node.add_incoming_edge(prev_cNode)

        for_cNode = self.create_cfg_node(node)
        for_cNode.add_incoming_edge(entry_node)
        last_cNode = self.connect_node_list(
                for_cNode, node.body, for_cNode, exit_node)

        if last_cNode is not None:
            for_cNode.add_incoming_edge(last_cNode)

        exit_node.add_incoming_edge(for_cNode)

        return exit_node

    def visit_While(self, node, prev_cNode, entry, exit):
        entry_node = CFGNode("ENTRY")
        exit_node = CFGNode("EXIT")

        entry_node.add_incoming_edge(prev_cNode)

        while_cNode = self.create_cfg_node(node)
        while_cNode.add_incoming_edge(entry_node)
        last_cNode = self.connect_node_list(
                while_cNode, node.body, while_cNode, exit_node)

        if last_cNode is not None:
            while_cNode.add_incoming_edge(last_cNode)

        exit_node.add_incoming_edge(while_cNode)

        return exit_node

    def visit_If(self, node, prev_cNode, entry, exit):
        entry_node = CFGNode("IF")
        entry_node.add_incoming_edge(prev_cNode)

        if_cNode = self.create_cfg_node(node)
        if_cNode.add_incoming_edge(entry_node)

        exit_cNode = CFGNode("FI")

        last_then_cNode = self.connect_node_list(
                if_cNode, node.body, entry, exit)

        if len(node.orelse) > 0:
            last_else_cNode = self.connect_node_list(
                    if_cNode, node.orelse, entry, exit)
        else:
            last_else_cNode = CFGNode("ELSE_PASS")
            last_else_cNode.add_incoming_edge(if_cNode)

        if last_then_cNode is not None:
            exit_cNode.add_incoming_edge(last_then_cNode)

        if last_else_cNode is not None:
            exit_cNode.add_incoming_edge(last_else_cNode)

        return exit_cNode

    def visit_Break(self, node, prev_cNode, entry, exit):
        cNode = self.create_cfg_node(node)
        cNode.add_incoming_edge(prev_cNode)
        exit.add_incoming_edge(cNode)
        return None

    def visit_Continue(self, node, prev_cNode, entry, exit):
        cNode = self.create_cfg_node(node)
        cNode.add_incoming_edge(prev_cNode)
        entry.add_incoming_edge(cNode)
        return None
