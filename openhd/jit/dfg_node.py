"""
========
DFG Node
========

This has the DFG node built by DFGbuilder as a tree structure
"""
import ast

import pydot
from PIL import Image

from ..dev import debug

TAG = "jit.dfg_node"

class DFGNode(object):
    """
    A node in the data flow graph
    Note: This is created in dfg_builder, and should be initialized with
    the factory function: create_dfg_node()
    """
    def __init__(self, aNode):
        self.aNode = aNode # AnnotationNode or name(str)

        self.incoming_edges = []
        self.outgoing_edges = []
        self.is_updapted_var = False

        self.chunk = None # This is set in the planner

    def add_incoming_edge(self, node, edge_tag=""):
        """ Add incoming edge in DFG builder """
        # link for both sides
        self.incoming_edges.append(node)
        node.outgoing_edges.append(self)

    def set_chunk(self, chunk):
        """ Set the chunk from the extract_maximum_trees """
        self.chunk = chunk

    # Utility functions
    def retrieve_var_name(self):
        """ return the name of the variable if it is """
        if isinstance(self.aNode, str):
            return self.aNode
        elif isinstance(self.aNode.item, ast.Name):
            return self.aNode.item.id

        return None

    def visualize_debug(self):
        VisualizeDFGNode().draw(self)

class VisualizeDFGNode(object):
    """
    For Debugging purpose - draw the DFG into a dot & png file
    """
    def __init__(self):
        self.all_edges = set() # set of (dNode, dNode)
        self.visited = set()

        self.gp = pydot.Dot(graph_type='digraph',fontname="Verdana")
        self.chunk_to_subgp = dict()

    def draw(self, dNode):
        self.visit(dNode)
        #print(self.all_edges)

        def get_nice_name_and_type(_dNode):
            if isinstance(_dNode.aNode, str):
                return _dNode.aNode, ast.Name
            elif isinstance(_dNode.aNode.item, ast.Name):
                return _dNode.aNode.item.id, ast.Name
            elif isinstance(_dNode.aNode.item, ast.Call):
                if isinstance(_dNode.aNode.item.func, ast.Attribute):
                    fname = _dNode.aNode.item.func.value.id + '.' + \
                            _dNode.aNode.item.func.attr
                else:
                    fname = _dNode.aNode.item.func.id

                return fname + "()", ast.Call
            elif isinstance(_dNode.aNode.item, str):
                return _dNode.aNode.item, str
            
            return str(type(_dNode.aNode.item)), type(_dNode.aNode.item)
            #return str(_dNode.aNode.item), type(_dNode.aNode.item)

        def add_dot_node(dNode):
            d_name, d_type = get_nice_name_and_type(dNode)

            dot_node = pydot.Node(str(id(dNode)), label=d_name)
            dot_node.set_style("filled")
            if d_type == ast.Name:
                if dNode.is_updapted_var:
                    dot_node.set("shape", "note")
                    dot_node.set_fillcolor("#eac3cf")
                else:
                    dot_node.set_style("\"filled,dashed\"")
                    dot_node.set("shape", "note")
                    dot_node.set_fillcolor("#f4e2e7")
            elif d_type == ast.Call:
                dot_node.set("shape", "box3d")
                dot_node.set_fillcolor("#cfe1e3")
            else:
                dot_node.set_fillcolor("#f2f3e6")

            # Add to the graph
            if dNode.chunk is None:
                self.gp.add_node(dot_node)
            else:
                if dNode.chunk not in self.chunk_to_subgp:
                    # Create subgraphs
                    cluster_name = str(hex(id(dNode.chunk.anode_items[0])))
                    subgp = pydot.Cluster(cluster_name, label=cluster_name)
                    self.gp.add_subgraph(subgp)
                    self.chunk_to_subgp[dNode.chunk] = subgp

                self.chunk_to_subgp[dNode.chunk].add_node(dot_node)
                
        for (dNode_in, dNode_out) in self.all_edges:
            add_dot_node(dNode_in)
            add_dot_node(dNode_out)
            self.gp.add_edge(pydot.Edge(str(id(dNode_in)), str(id(dNode_out))))

        #FILENAME = "debug_dfg"
        #self.gp.write_raw(FILENAME + '.dot')
        #self.gp.write_png(FILENAME + '.png')
        #debug.log(debug.DEV, TAG, "DFG is stored in %s.dot & .png" % FILENAME)

        #im=Image.open(FILENAME + '.png')
        #im.show()

    def visit(self, dNode):
        if dNode in self.visited:
            return
        
        self.visited.add(dNode)

        for dNode_in in dNode.incoming_edges:
            self.visit(dNode_in)
            self.all_edges.add((dNode_in, dNode))

        for dNode_out in dNode.outgoing_edges:
            self.visit(dNode_out)
            self.all_edges.add((dNode, dNode_out))

