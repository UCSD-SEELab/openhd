"""
===============
DFG builder
===============

This builds the data flow graph
"""

import ast

from ..dev import debug

from .dfg_node import DFGNode

TAG = "jit.dfg_builder"

def build_data_flow_graph(annotation_root, arg_variables):
    """ Build the data flow graph"""
    builder = DFGBuilder(arg_variables)
    aNode_body = unshell(annotation_root)

    dfg_body = builder.visit(aNode_body)
    #dfg_body.visualize_debug()

    return dfg_body, builder.aNode_to_dNode

def unshell(annotation_root):
    """ Return the body of the function """
    aNode_func = annotation_root.child[0].child[0]
    aNode_body = None

    for subchild in aNode_func.child:
        if subchild.item == "body":
            aNode_body = subchild
            break

    assert(aNode_body)
    return aNode_body

class DFGBuilder(object):
    def __init__(self, arg_variables):
        self.updated_var_to_dNode = dict() # name(str) -> dNode
        self.aNode_to_dNode = dict() # aNode or name(str) -> dNode

        for n, (t, _) in arg_variables.items():
            dNode = self.create_dfg_node(n)
            self.track_to_updated_var(n, dNode)

    def create_dfg_node(self, aNode):
        # aNode can be a string if it's an arg variable
        dNode = DFGNode(aNode) 
        self.aNode_to_dNode[aNode] = dNode
        return dNode

    def track_to_updated_var(self, name, dNode):
        self.updated_var_to_dNode[name] = dNode
        dNode.is_updapted_var = True

    def visit(self, aNode):
        """Visit an annotated node."""
        method = 'visit_' + aNode.item.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        dNode = visitor(aNode)
        return dNode

    def generic_visit(self, aNode):
        dNode = self.create_dfg_node(aNode)

        if aNode.child is None:
            return dNode

        if isinstance(aNode.child, list):
            for subchild in aNode.child:
                child_dNode = self.visit(subchild)
                if child_dNode is not None:
                    dNode.add_incoming_edge(child_dNode)
        else:
            child_dNode = self.visit(aNode.child)
            if child_dNode is not None:
                dNode.add_incoming_edge(child_dNode)

        return dNode

    boolean_constants = frozenset(['True', 'False'])
    def visit_Name(self, aNode):
        dNode = self.create_dfg_node(aNode)

        name = aNode.item.id
        if name in self.boolean_constants:
            return dNode

        debug.thrifty_assert(
                name in self.updated_var_to_dNode,
                "Cannot find the dependency: %s. " % name + \
                "Is it really declared somewhere?")

        dNode.add_incoming_edge(self.updated_var_to_dNode[name])
        return dNode

    def visit_Assign(self, aNode):
        dNode = self.create_dfg_node(aNode)
        assert(len(aNode.child) == 2)
        
        var_aNode = aNode.child[0]
        val_aNode = aNode.child[1]

        if isinstance(var_aNode.item, ast.Tuple):
            # Currently all pairs are merged into an assign node
            for subvar, subval in zip(var_aNode.child, val_aNode.child):
                assert(isinstance(subvar.item, ast.Name))
                var_dNode = self.create_dfg_node(subvar)
                val_dNode = self.visit(subval)

                dNode.add_incoming_edge(val_dNode)
                var_dNode.add_incoming_edge(dNode)

                self.track_to_updated_var(subvar.item.id, var_dNode)
        elif isinstance(var_aNode.item, ast.Name):
            var_dNode = self.create_dfg_node(var_aNode)
            val_dNode = self.visit(val_aNode)

            dNode.add_incoming_edge(val_dNode)
            var_dNode.add_incoming_edge(dNode)

            self.track_to_updated_var(var_aNode.item.id, var_dNode)
        elif isinstance(var_aNode.item, ast.Subscript):
            var_dNode = self.visit(var_aNode)
            val_dNode = self.visit(val_aNode)

            dNode.add_incoming_edge(val_dNode)
            var_dNode.add_incoming_edge(dNode)

            self.track_to_updated_var(
                    var_aNode.child[0].item.id, 
                    var_dNode.incoming_edges[0])
        else:
            raise NotImplementedError()

        return dNode

    def visit_AugAssign(self, aNode):
        dNode = self.create_dfg_node(aNode)
        assert(len(aNode.child) == 2)

        var_aNode = aNode.child[0]
        val_aNode = aNode.child[1]

        if isinstance(var_aNode.item, ast.Name):
            name = var_aNode.item.id

            var_dNode = self.create_dfg_node(var_aNode)
            val_dNode = self.visit(val_aNode)
            dNode.add_incoming_edge(val_dNode)
            dNode.add_incoming_edge(self.updated_var_to_dNode[name])
            var_dNode.add_incoming_edge(dNode)

            self.track_to_updated_var(name, var_dNode)
        elif isinstance(var_aNode.item, ast.Subscript):
            name = var_aNode.child[0].item.id

            var_dNode = self.visit(var_aNode)
            val_dNode = self.visit(val_aNode)
            dNode.add_incoming_edge(val_dNode)
            dNode.add_incoming_edge(self.updated_var_to_dNode[name])
            var_dNode.add_incoming_edge(dNode)

            self.track_to_updated_var(name, var_dNode.incoming_edges[0])
        else:
            assert(False)

        return dNode

    def visit_Subscript(self, aNode):
        dNode = self.create_dfg_node(aNode)

        var_aNode = aNode.child[0]
        index_aNode = aNode.child[1]

        var_dNode = self.visit(var_aNode)
        index_dNode = self.visit(index_aNode)
        dNode.add_incoming_edge(var_dNode)
        dNode.add_incoming_edge(index_dNode)

        return dNode

    def visit_For(self, aNode):
        dNode = self.create_dfg_node(aNode)
        assert(len(aNode.child) == 3)
        var_aNode = aNode.child[0]
        var_dNode = self.create_dfg_node(var_aNode)
        self.track_to_updated_var(var_aNode.item.id, var_dNode)

        dNode.add_incoming_edge(self.visit(aNode.child[1]))
        dNode.add_incoming_edge(self.visit(aNode.child[2]))
        var_dNode.add_incoming_edge(dNode)

        return dNode

