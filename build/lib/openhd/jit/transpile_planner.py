"""
=================
Transpile Planner
=================

This plans which parts should be compiled on CUDA side
and identifies relavant variable informations, e.g., inputs, outputs, allocs,
"""
import ast

from ..dev import debug

from ..jit import HYPERVECTOR_TYPE, HYPERMATRIX_TYPE
from ..jit import NP_INT_ARRAY_TYPE, NP_FLOAT_ARRAY_TYPE
from ..jit import HD_INT_ARRAY_TYPE, HD_FLOAT_ARRAY_TYPE

TAG = "jit.transpile_planner"

def plan_transpile(scanned_parcel, arg_variables, encode_func_vars):
    # Identify impossible edge
    impossible_edges = identify_impossible_edge(scanned_parcel.aNode_to_dNode)

    # Find the maximum tree
    chunk_list = extract_maximum_trees(scanned_parcel.aRoot, impossible_edges)

    # Mark the dfg with chunk
    for chunk in chunk_list:
        mark_chunk_to_dfg(
                chunk, chunk.anode_items,
                scanned_parcel.aNode_to_dNode)

    # Post analysis with the marked DFG
    for chunk in chunk_list:
        update_used_var_types(chunk)
        replace_hd_array_types(chunk)
        update_var_flows(chunk, scanned_parcel.aNode_to_dNode, arg_variables)

        if encode_func_vars is not None:
            chunk.outputs.add(encode_func_vars[1])


        #chunk.print_debug()

    scanned_parcel.dBody.visualize_debug()

    return chunk_list


class CUDAableChunk(object):
    """
    This includes a chunk that can be converted to a CUDA kernel
    """
    def __init__(self, anode_items):
        self.anode_items = anode_items

        # Variables set in the post analysis:
        # update_used_var_types & update_var_flows
        self.used_var_types = dict() 
        self.consts = set()
        self.inputs = set()
        self.outputs = set()
        self.allocs = set()
        self.launcher_func_name = None
        self.hd_arrays = set()

    def get_launcher_func_name(self):
        """
        Generate the launcher function (in Python)
        with the first ast ID in the chunk
        """
        if not self.launcher_func_name:
            # We create and save it as a member variable
            # since AST is invalidated in the cuda code transpile
            self.launcher_func_name = \
                    'OPENHD_CALL_' + str(hex(id(self.anode_items[0].item)))
        return self.launcher_func_name


    def __str__(self):
        return str(self.anode_items)

    def print_debug(self):
        debug.log(debug.DEV, TAG, str(self.anode_items) + " : ")
        debug.log(debug.DEV, TAG, str(self.used_var_types))
        debug.log(debug.DEV, TAG, str(self.consts))
        debug.log(debug.DEV, TAG, str(self.inputs))
        debug.log(debug.DEV, TAG, str(self.outputs))
        debug.log(debug.DEV, TAG, str(self.allocs))


def extract_maximum_trees(aNode, impossible_edges):
    """
    Extract the list of CUDAableChunk 
    """
    # This function, extract_maximum_trees, is a recursive call

    def is_transformable_ast(aNode):
        if aNode.transformable == False:
            return False

        return any(isinstance(aNode.item, ast_type) for ast_type in [
                            ast.Assign,
                            ast.AugAssign,
                            ast.While,
                            ast.If,
                            ast.For,
                            ast.Call,
                            ast.Expr
                            ])

    def is_possible_ast_list(aNode_list, impossible_edges):
        if len(impossible_edges) == 0:
            return True

        flattens = []
        for aNode in aNode_list:
            flattens += aNode.flatten()

        for (aNode0, aNode1) in impossible_edges:
            if aNode0 in flattens and aNode1 in flattens:
                return False

        return True


    # Run it after visiting + check the openhd inclusion
    # - It uses breath-first search
    if is_possible_ast_list([aNode], impossible_edges) and \
            is_transformable_ast(aNode) and aNode.openhd_included:
        return [CUDAableChunk([aNode])]

    if aNode.child is None:
        return None

    if not isinstance(aNode.child, list):
        return extract_maximum_trees(aNode.child, impossible_edges)

    # Spread the extractable nodes
    must_extract_children = [-1] * len(aNode.child)
    for idx, c in enumerate(aNode.child):
        if is_possible_ast_list([c], impossible_edges) and \
                is_transformable_ast(c) and c.openhd_included:
            must_extract_children[idx] = idx
            spreaded = [c]

            # upper sweep
            for subidx in range(idx-1, -1, -1):
                cc = aNode.child[subidx]
                if is_possible_ast_list(spreaded + [cc], impossible_edges) and \
                        is_transformable_ast(cc):
                    must_extract_children[subidx] = idx
                    spreaded.append(cc)
                else:
                    break

            # down sweep
            for subidx in range(idx, len(aNode.child)):
                cc = aNode.child[subidx]
                if is_possible_ast_list(spreaded + [cc], impossible_edges) and \
                        is_transformable_ast(cc):
                    must_extract_children[subidx] = idx
                    spreaded.append(cc)
                else:
                    break

    # Create the chunk list
    chunk_list = []
    anode_items = []
    prev_m = -1
    for m, c in zip(must_extract_children, aNode.child):
        if prev_m != m:
            if len(anode_items) > 0:
                chunk_list.append(CUDAableChunk(anode_items))
                anode_items = []

        prev_m = m

        if m == -1:
            continue

        anode_items.append(c)

    if len(anode_items) > 0:
        chunk_list.append(CUDAableChunk(anode_items))

    # Go into the one-level more for the items that's not included
    for m, c in zip(must_extract_children, aNode.child):
        if m != -1:
            continue

        ret = extract_maximum_trees(c, impossible_edges)
        if ret is not None:
            chunk_list += ret

    return chunk_list

def update_used_var_types(chunk):
    """
    update var types for a chunk
    """
    def update_type(used_var_types, name, item_type):
        def is_either_type(item1_type, item2_type, item_type):
            if item1_type == item_type:
                return True

            if item2_type == item_type:
                return True

            return False

        if name not in used_var_types:
            used_var_types[name] = item_type
            return

        # Check previous type
        prev_type = used_var_types[name]
        if prev_type is None:
            used_var_types[name] = item_type
        elif prev_type == item_type: 
            return

        # Perform type inference for the variable name
        if is_either_type(prev_type, item_type, HYPERMATRIX_TYPE) or \
                is_either_type(prev_type, item_type, HYPERVECTOR_TYPE):
            debug.thrifty_assert(
                    prev_type == item_type,
                    "Dynamic type conversion is not allowed " + \
                            "for hypervector types") 

        primitive_type_order = [float, int, bool]
        for t in primitive_type_order:
            if is_either_type(prev_type, item_type, t):
                used_var_types[name] = t
                return

    def _traverse_tree(aNode, chunk): # aNode could be a list
        if isinstance(aNode, list):
            for c in aNode:
                _traverse_tree(c, chunk)
            return

        if isinstance(aNode.item, ast.Name):
            update_type(chunk.used_var_types, aNode.item.id, aNode.item_type)
                        
        if aNode.child is None:
            return 

        _traverse_tree(aNode.child, chunk)

    _traverse_tree(chunk.anode_items, chunk)

def mark_chunk_to_dfg(chunk, aNode, aNode_to_dNode):
    """
    Recursively mark the DFG node with the chunk.
    aNode can be a list of annotated nodes
    """
    if isinstance(aNode, list):
        for c in aNode:
            mark_chunk_to_dfg(chunk, c, aNode_to_dNode)
        return

    if aNode in aNode_to_dNode:
        aNode_to_dNode[aNode].set_chunk(chunk)
                    
    # Recursively run for children if any
    if aNode.child is None:
        return 

    mark_chunk_to_dfg(chunk, aNode.child, aNode_to_dNode)

def replace_hd_array_types(chunk):
    """
    Change HD_INT_ARRAY_TYPE or HD_FLOAT_ARRAY_TYPE to corresponding np array
    types, and add to hd_arrays so that cuda launching code can handle them
    without changing the access fusion procedure.
    """
    for var in chunk.used_var_types:
        var_type = chunk.used_var_types[var]
        if var_type == HD_FLOAT_ARRAY_TYPE:
            chunk.used_var_types[var] = NP_FLOAT_ARRAY_TYPE
            chunk.hd_arrays.add(var)
        elif var_type == HD_INT_ARRAY_TYPE:
            chunk.used_var_types[var] = NP_INT_ARRAY_TYPE
            chunk.hd_arrays.add(var)

def update_var_flows(chunk, aNode_to_dNode, arg_variables):
    """
    Update variable flows in a chunk to get consts, inputs, outputs, updates
    with the data flow graph
    """

    # Const: We include all variables in constant-able types,
    # and will remove one appeared in DFG 
    chunk.consts = set() 
    for var in chunk.used_var_types:
        if chunk.used_var_types[var] in [int, float, bool] and \
                var in arg_variables:
            # Skip encoder-specific variable
            if var == "__stream__": 
                continue
            if var == "__base_n__": 
                continue

            chunk.consts.add(var)

    
    visited = set()
    def traverse_graph(dNode, this_chunk):
        if dNode in visited:
            return
        visited.add(dNode)

        # Loop for incoming edges
        for dNode_in in dNode.incoming_edges:
            if dNode_in.chunk == this_chunk:
                traverse_graph(dNode_in, this_chunk)
            else:
                # Input: any incoming variable located outside of this chunk
                name = dNode_in.retrieve_var_name()
                if name is not None:
                    this_chunk.inputs.add(name)

        # Loop for outgoing edges
        name = dNode.retrieve_var_name()
        for dNode_out in dNode.outgoing_edges:
            if dNode_out.chunk == this_chunk:
                traverse_graph(dNode_out, this_chunk)
            else:
                # Output: any outgoing varaible to somewhere outside of this  
                if name is not None:
                    this_chunk.outputs.add(name)

        if name is not None and dNode.is_updapted_var: 
            # Const
            this_chunk.consts -= set([name])

            # Alloc
            if dNode.aNode.alloc_hv:
                this_chunk.allocs.add(name)

    for aNode in chunk.anode_items:
        traverse_graph(aNode_to_dNode[aNode], chunk)


def identify_impossible_edge(aNode_to_dNode):
    def add_incoming(dNode, visited):
        if dNode in visited:
            return []
        visited.add(dNode)

        if not isinstance(dNode.aNode, str):
            if isinstance(dNode.aNode.item, ast.Name):
                # Add all edges for the hypervector types
                if dNode.aNode.item_type == HYPERVECTOR_TYPE or \
                       dNode.aNode.item_type == HYPERMATRIX_TYPE:

                    return [(
                            dNode.incoming_edges[0].aNode,
                            dNode.aNode
                            )]

        impossible_edges = []
        for dNode_in in dNode.incoming_edges:
            impossible_edges += add_incoming(dNode_in, visited)

        return impossible_edges

    def add_outgoing(dNode, fname): # fname is given for the error message
        assign_aNode = dNode.outgoing_edges[0].aNode
        debug.thrifty_assert(
                isinstance(assign_aNode.item, ast.Assign),
                "Function %s has to be directly used for assignment." % fname)
        list_node = assign_aNode.parent
        assign_node_idx = list_node.child.index(assign_aNode)
        if assign_node_idx == len(list_node.child) - 1:
            # Terminate to the parent (e.g., for loop) 
            return [(assign_aNode, assign_aNode.parent)]
        else:
            # Terminate to the next statement (e.g., for assign inside body) 
            return [(assign_aNode, list_node.child[assign_node_idx + 1])]


    # 1. Permute cannot happen in a single chunk across load / write
    # 2. Shuffle must terminate all outgoings in the chunk
    # NOTE: 3. Search, similarity, etcs terminates both incoming and outgoings
    # So, currently they are excluded from JIT
    incoming_terminators = ["permute"] #, "search"]
    outgoing_terminators = ["shuffle"] #, "search"]
    impossible_edges = []
    visited = set()
    for aNode, dNode in aNode_to_dNode.items():
        if isinstance(aNode, str):
            continue
        if not isinstance(aNode.item, ast.Call):
            continue
        if not isinstance(aNode.item.func, ast.Attribute):
            continue

        fname = aNode.item.func.attr
        if fname in incoming_terminators:
            # Find the impossible edge due to permute
            impossible_edges += add_incoming(dNode, visited)
        elif fname in outgoing_terminators:
            # Find the impossible edge due to shuffle 
            impossible_edges += add_outgoing(dNode, fname)

    return impossible_edges
