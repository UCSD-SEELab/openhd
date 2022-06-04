"""
========
CFG Node
========

This has the CFG node built by CFGbuilder as a tree structure
"""
import ast

import pydot
from PIL import Image

from ..dev import debug

TAG = "jit.cfg_node"

class CFGNode(object):
    """
    A node in the control flow graph
    Note: This is created in cfg_builder, and should be initialized with
    the factory function: create_cfg_node()
    """
    def __init__(self, node):
        self.node = node    # ast node or string
                            # (START, END, FI, ELSE_PASS, ENTRY, EXIT)

        self.incoming_edges = []
        self.outgoing_edges = []

    def add_incoming_edge(self, node):
        """ Add incoming edge in CFG builder """
        # link for both sides
        self.incoming_edges.append(node)
        node.outgoing_edges.append(self)

    def visualize_debug(self):
        VisualizeCFGNode().draw(self)

class VisualizeCFGNode(object):
    """
    For Debugging purpose - draw the DFG into a dot & png file
    """
    def __init__(self):
        self.all_edges = set() # set of (cNode, cNode)
        self.visited = set()

        self.gp = pydot.Dot(graph_type='digraph',fontname="Verdana")
        self.chunk_to_subgp = dict()

    def draw(self, cNode):
        self.visit(cNode)

        def get_nice_name_and_type(_cNode):
            if isinstance(_cNode.node, str):
                return _cNode.node, str
            
            return str(type(_cNode.node)), type(_cNode.node)

        def add_dot_node(cNode):
            c_name, c_type = get_nice_name_and_type(cNode)

            dot_node = pydot.Node(str(id(cNode)), label=c_name)
            dot_node.set_style("filled")
            if c_type == str:
                if c_name == "IF": 
                    dot_node.set_style("\"filled,dashed\"")
                    dot_node.set("shape", "diamond")
                    dot_node.set_fillcolor("#cfe1e3")
                elif c_name == "START" or c_name == "END":
                    dot_node.set_style("\"filled,dashed\"")
                    dot_node.set("shape", "box3d")
                    dot_node.set_fillcolor("#eac3cf")
                else: # ELSE_PASS, FI, ENTRY, EXIT
                    dot_node.set_style("\"filled,dashed\"")
                    dot_node.set("shape", "box")
                    dot_node.set_fillcolor("#cfe1e3")
            else: # AST
                dot_node.set_fillcolor("#f2f3e6")

            self.gp.add_node(dot_node)

        for (cNode_in, cNode_out) in self.all_edges:
            add_dot_node(cNode_in)
            add_dot_node(cNode_out)
            self.gp.add_edge(pydot.Edge(str(id(cNode_in)), str(id(cNode_out))))

        #FILENAME = "debug_cfg"
        #self.gp.write_raw(FILENAME + '.dot')
        #self.gp.write_png(FILENAME + '.png')
        #debug.log(debug.DEV, TAG, "CFG is stored in %s.dot & .png" % FILENAME)

        #im=Image.open(FILENAME + '.png')
        #im.show()

    def visit(self, cNode):
        if cNode in self.visited:
            return
        
        self.visited.add(cNode)

        for cNode_in in cNode.incoming_edges:
            self.visit(cNode_in)
            self.all_edges.add((cNode_in, cNode))

        for cNode_out in cNode.outgoing_edges:
            self.visit(cNode_out)
            self.all_edges.add((cNode, cNode_out))

