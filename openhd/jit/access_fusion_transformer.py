"""
=============
Access fusion
=============

This fuses the global memory accesses of hypervector/matrix variables in a chunk
to reduce the amount of the global memory access
"""

import ast
import hashlib
from ast import iter_fields, copy_location
from collections import defaultdict

from ..dev import debug

from ..jit import jit_codegen as codegen
from ..jit import HYPERVECTOR_TYPE, HYPERMATRIX_TYPE, HYPERELEMNT_TYPE
from ..jit import ARG_PREFIX, STRIDE_POSTFIX
from ..jit import NP_INT_ARRAY_TYPE, NP_FLOAT_ARRAY_TYPE
from ..jit import HD_INT_ARRAY_TYPE, HD_FLOAT_ARRAY_TYPE

from .cfg_builder import build_control_flow_graph

TAG = "jit.access_fusion_transformer"


def fuse_access(node, chunk):
    """ Fuse IO """
    # Build control flow graph
    cfg_root = build_control_flow_graph(node)

    # Decide fusion accesses with CFG
    fusionIdentifier = FusionIdentifier(cfg_root, chunk.used_var_types)
    fusionIdentifier.run()

    #print(fusionIdentifier.updated_vars)
    #print("READ", fusionIdentifier.reads_in_node)
    #print("WRITE", fusionIdentifier.writes_in_node)
    #print("MULTI", fusionIdentifier.multi_access_nodes)
    #print(ast.dump(node))

    # Transform the node using the CFG
    new_node = AccessFusionTransformer(fusionIdentifier).visit(node)
    new_node = RegisterNameTransformer(fusionIdentifier).visit(new_node)

    #print(ast.dump(new_node))
    #debug.log(debug.DEV, TAG, codegen.to_source(new_node), add_lineno=True)
    #assert(False)

    # update new names in types
    for _, name in fusionIdentifier.node_to_register_name.items():
        org_name = fusionIdentifier.register_name_to_original_name[name]

        if org_name in chunk.used_var_types and \
                chunk.used_var_types[org_name] == NP_INT_ARRAY_TYPE:
            chunk.used_var_types[name] = int 
        elif org_name in chunk.used_var_types and \
                chunk.used_var_types[org_name] == NP_FLOAT_ARRAY_TYPE:
            chunk.used_var_types[name] = float 
        elif org_name in chunk.used_var_types and \
                chunk.used_var_types[org_name] == HD_INT_ARRAY_TYPE:
            chunk.used_var_types[name] = int 
        elif org_name in chunk.used_var_types and \
                chunk.used_var_types[org_name] == HD_FLOAT_ARRAY_TYPE:
            chunk.used_var_types[name] = float 
        else:
            # The type will be inferred during data type mutation
            chunk.used_var_types[name] = HYPERELEMNT_TYPE 

    return new_node


class FusionIdentifier(object):
    def __init__(self, cfg_root, used_var_types):
        self.cfg_root = cfg_root
        self.used_var_types = used_var_types
        self.traversed = set()
        self.now_traverse_cNode = None

        # Identified vars for step 1
        self.updated_vars = defaultdict(list)
        self.write_trip_vars = defaultdict(list)
        self.read_trip_vars = defaultdict(list)
        self.node_to_register_name = dict()
        self.node_to_replaced_ast = dict()
        self.register_name_to_original_name = dict()
        self.shuffled_write = dict() # key -> ast to overwrite

        # Identified vars for step 2
        self.key_to_replaced_ast = dict()
        self.writes_in_node = defaultdict(dict)
        self.reads_in_node = defaultdict(dict)
        self.multi_access_nodes = set()

    def run(self):
        # Step 1: Find the updated_vars, write_trip_vars, read_trip_vars
        self.traverse(self.cfg_root)

        # Step 2: Find the variables needed to be fused
        self.decide_fusion()

    def decide_fusion(self):
        # Propagate write 
        for cNode in self.write_trip_vars:
            for node, name, dependents, replaced_ast in \
                    self.write_trip_vars[cNode]:
                self.traversed = set()
                self.decide_write_fusion(
                        cNode, cNode, node, name, dependents, replaced_ast)

        # Propagate read 
        for cNode in self.read_trip_vars:
            for node, name, dependents, replaced_ast in \
                    self.read_trip_vars[cNode]:
                self.traversed = set()
                self.decide_read_fusion(
                        cNode, cNode, node, name, dependents, replaced_ast)

        # Find multi-access nodes
        for _node in self.writes_in_node:
            for _key in self.writes_in_node[_node]:
                if len(self.writes_in_node[_node][_key]) > 1:
                    for var_node in self.writes_in_node[_node][_key]:
                        self.multi_access_nodes.add(var_node)

        for _node in self.reads_in_node:
            for _key in self.reads_in_node[_node]:
                if len(self.reads_in_node[_node][_key]) > 1:
                    for var_node in self.reads_in_node[_node][_key]:
                        self.multi_access_nodes.add(var_node)


    def create_key(self, name, replaced_ast):
        ret = name + "##" 
        ret += codegen.to_source(replaced_ast)

        self.key_to_replaced_ast[ret] = (name, replaced_ast)

        return ret

    def decide_write_fusion(
            self, cNode, cNode_start, node, name, dependents, replaced_ast):
        # Traverse outgoing edges
        if cNode == cNode_start and cNode in self.traversed:
            # revisited -> multiple write accesses
            self.multi_access_nodes.add(node)
            return

        if cNode in self.traversed:
            return

        self.traversed.add(cNode)
        key = self.create_key(name, replaced_ast)
        #print(key, cNode_start.node, cNode.node)

        # check if any dependent var is updated
        if cNode != cNode_start:
            if dependents and (cNode in self.updated_vars):
                for d in dependents:
                    if d in self.updated_vars[cNode]:
                        if key not in self.writes_in_node[cNode.node]:
                            self.writes_in_node[cNode.node][key] = []
                        self.writes_in_node[cNode.node][key].append(node)
                        return # No more propagating

        # check if it's the end node
        if cNode.node == "END":
            if key not in self.writes_in_node[cNode.node]:
                self.writes_in_node[cNode.node][key] = []
            self.writes_in_node[cNode.node][key].append(node)
            return

        # No depednecy found
        for cNode_out in cNode.outgoing_edges:
            self.decide_write_fusion(
                    cNode_out, cNode_start, node, name,
                    dependents, replaced_ast)

    def decide_read_fusion(
            self, cNode, cNode_start, node, name, dependents, replaced_ast):
        # Traverse outgoing edges
        # It is really similar to the write fusion, but checks 
        # overlapped writing
        if cNode == cNode_start and cNode in self.traversed:
            # revisited -> multiple write accesses. add one more
            self.multi_access_nodes.add(node)
            return

        if cNode in self.traversed:
            return

        self.traversed.add(cNode)
        key = self.create_key(name, replaced_ast)
        #print(key, cNode_start.node, cNode.node)

        if cNode != cNode_start:
            # check if any dependent var is updated
            if dependents and cNode in self.updated_vars:
                for d in dependents:
                    if d in self.updated_vars[cNode]:
                        if key not in self.reads_in_node[cNode.node]:
                            self.reads_in_node[cNode.node][key] = []
                        self.reads_in_node[cNode.node][key].append(node)
                        return # No more propagating

            # check if the current key is written
            if cNode in self.write_trip_vars:
                for _node, _name, __, _replaced_ast in \
                        self.write_trip_vars[cNode]:
                    if key == self.create_key(_name, _replaced_ast):
                        self.multi_access_nodes.add(_node)
                        self.multi_access_nodes.add(node)
                        return # No more propagating

            # check if __d__ is met
            if self.used_var_types[name] == HYPERVECTOR_TYPE or \
                    self.used_var_types[name] == HYPERMATRIX_TYPE and \
                    "__d__" in self.updated_vars[cNode]:
                if key not in self.reads_in_node[cNode.node]:
                    self.reads_in_node[cNode.node][key] = []
                self.reads_in_node[cNode.node][key].append(node)
                return # No more propagating


        # check if it's the end node
        if cNode.node == "START":
            if key not in self.reads_in_node[cNode.node]:
                self.reads_in_node[cNode.node][key] = []
            self.reads_in_node[cNode.node][key].append(node)
            return

        # No depednecy found
        for cNode_in in cNode.incoming_edges:
            self.decide_read_fusion(
                    cNode_in, cNode_start, node, name,
                    dependents, replaced_ast)


    def traverse(self, cNode):
        if cNode in self.traversed:
            return
        
        self.traversed.add(cNode)
        #print(cNode, cNode.node)

        # Identify the used variables
        if not isinstance(cNode.node, str): 
            self.now_traverse_cNode = cNode
            self.visit(cNode.node) # for ast

        # Only visit outgoing edges
        for cNode_out in cNode.outgoing_edges:
            self.traverse(cNode_out)

    def visit(self, node):
        # node: ast node / return: list of var names dependent
        """ Visit an AST node ."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        if len(node._fields) == 0:
            return set([])

        dependents = set()
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        dependents |= self.visit(item)
            elif isinstance(value, ast.AST):
                dependents |= self.visit(value)

        return dependents

    def is_name_array(self, name):
        if name not in self.used_var_types:
            return False
        return self.used_var_types[name] == HYPERVECTOR_TYPE or \
                self.used_var_types[name] == HYPERMATRIX_TYPE or \
                self.used_var_types[name] == NP_FLOAT_ARRAY_TYPE or \
                self.used_var_types[name] == NP_INT_ARRAY_TYPE

    def is_name_hv_array(self, name):
        if name not in self.used_var_types:
            return False
        return self.used_var_types[name] == HYPERVECTOR_TYPE or \
                self.used_var_types[name] == HYPERMATRIX_TYPE

    def is_name_np_array(self, name):
        if name not in self.used_var_types:
            return False
        return self.used_var_types[name] == NP_FLOAT_ARRAY_TYPE or \
                self.used_var_types[name] == NP_INT_ARRAY_TYPE

    @staticmethod
    def gen_hvdim(node):
        func = ast.Name("*__hvdim__", ast.Load())
        args = [
                ast.Name(ARG_PREFIX + node.id, ast.Load()),
                ast.Name("__d__", ast.Load())
                ]
        return ast.Call(func, args, [])
        # return ast.Call(func, args, [], None, None)

    def gen_arraydim(self, node):
        if self.is_name_np_array(node.value.id):
            if isinstance(node.slice.value, ast.Tuple):
                debug.thrifty_assert(
                        len(node.slice.value.elts) == 2,
                        "Numpy array dimsion has to be 1 or 2."
                        )
                func = ast.Name("*__npdim_2d__", ast.Load())
                args = [
                        ast.Name(ARG_PREFIX + node.value.id, ast.Load()),
                        node.slice.value.elts[0],
                        node.slice.value.elts[1],
                        ast.Name(ARG_PREFIX + node.value.id + \
                                STRIDE_POSTFIX, ast.Load()),
                        ]
            else:
                func = ast.Name("*__npdim__", ast.Load())
                args = [
                        ast.Name(ARG_PREFIX + node.value.id, ast.Load()),
                        node.slice.value
                        ]
        else:
            func = ast.Name("*__hxdim__", ast.Load())
            args = [
                    ast.Name(ARG_PREFIX + node.value.id, ast.Load()),
                    node.slice.value,
                    ast.Name("__d__", ast.Load())
                    ]

        return ast.Call(func, args, [])
        # return ast.Call(func, args, [], None, None)

    @staticmethod
    def gen_permute(node):
        func = ast.Name("*__permute__", ast.Load())
        args = [
                ast.Name(ARG_PREFIX + node.args[0].id, ast.Load()),
                node.args[1],
                ast.Name("__d__", ast.Load())
                ]
        return ast.Call(func, args, [])
        # return ast.Call(func, args, [], None, None)


    @staticmethod
    def gen_hv_shuffle(node, arg):
        func = ast.Name("*__hv_shuffle__", ast.Load())
        args = [
                ast.Name(ARG_PREFIX + node.id, ast.Load()),
                arg,
                ast.Name("__d__", ast.Load())
                ]
        return ast.Call(func, args, [])
        # return ast.Call(func, args, [], None, None)

    @staticmethod
    def gen_hx_shuffle(node, arg):
        func = ast.Name("*__hx_shuffle__", ast.Load())
        args = [
                ast.Name(ARG_PREFIX + node.value.id, ast.Load()),
                node.slice.value,
                arg,
                ast.Name("__d__", ast.Load())
                ]

        return ast.Call(func, args, [])
        # return ast.Call(func, args, [], None, None)

    def visit_Num(self, node):
        return set([node.n])

    def visit_While(self, node):
        # Nothing to return
        return set([])

    def visit_For(self, node):
        assert(isinstance(node.target, ast.Name))
        self.add_updated_var(node.target.id)
        return set([])

    def visit_If(self, node):
        self.visit(node.test)

        # Nothing to return
        return set([])

    def visit_Attribute(self, node):
        # TODO: Attribute variable???
        return set([])

    def visit_Name(self, node):
        if self.is_name_hv_array(node.id):
            self.add_read_trip(
                    node, node.id, None, FusionIdentifier.gen_hvdim(node))
        return set([node.id])

    def write_trip_lvalue(self, replaced_node, target):
        if isinstance(target, ast.Name):
            if self.is_name_hv_array(target.id):
                self.add_write_trip(
                        replaced_node, target.id,
                        None, FusionIdentifier.gen_hvdim(target))
            else:
                self.add_updated_var(target.id)
            return True
        elif isinstance(target, ast.Tuple):
            for subtarget in target.elts:
                # TODO: support hypermatrix in tuple
                if self.is_name_hv_array(subtarget.id):
                    self.add_write_trip(
                            subtarget, subtarget.id,
                            None, FusionIdentifier.gen_hvdim(subtarget))
                else:
                    self.add_updated_var(subtarget.id)
            return True
        elif isinstance(target, ast.Subscript):
            if self.is_name_array(target.value.id):
                self.add_write_trip(
                        replaced_node,
                        target.value.id,
                        self.visit(target.slice),
                        self.gen_arraydim(target))
            else:
                # no subscription supports for non-hv type
                assert(False)

            return True
        return None

    def visit_Assign(self, node):
        self.visit(node.value)

        if self.write_trip_lvalue(node.targets[0], node.targets[0]):
            pass
        elif isinstance(node.targets[0], ast.Call):
            call_node = node.targets[0]
            if isinstance(call_node.func, ast.Attribute):
                fname = call_node.func.attr # TODO: package name - node.func.value.id
            elif isinstance(call_node.func, ast.Name):
                fname = call_node.func.id

            debug.thrifty_assert(
                    fname == "shuffle",
                    "Function call cannot be used as l-value."
                    )

            var_node = call_node.args[0]
            arg_node = call_node.args[1]
            debug.thrifty_assert(
                    not isinstance(var_node, ast.Tuple),
                    "Not implemented: shuffle cannot be used with Tuple"
                    )

            debug.thrifty_assert(
                    self.write_trip_lvalue(node.targets[0], var_node),
                    "The first argument of shuffle must be a hypervector."
                    )

            if isinstance(var_node, ast.Name):
                key = self.create_key(
                        var_node.id, FusionIdentifier.gen_hvdim(var_node))
                self.shuffled_write[key] = FusionIdentifier.gen_hv_shuffle(
                        var_node, arg_node)
            elif isinstance(var_node, ast.Subscript):
                key = self.create_key(
                        var_node.value.id, self.gen_arraydim(var_node))
                self.shuffled_write[key] = FusionIdentifier.gen_hx_shuffle(
                        var_node, arg_node)
            else:
                # Not implemented
                assert(False)
        
        return set([])

    def visit_AugAssign(self, node):
        self.visit(node.value)

        if isinstance(node.target, ast.Name):
            node_name = node.target.id
            if self.is_name_hv_array(node_name):
                self.add_write_trip(
                        node.target, node_name,
                        None, FusionIdentifier.gen_hvdim(node.target))
                self.add_read_trip(
                        node.target, node_name,
                        None, FusionIdentifier.gen_hvdim(node.target))
            else:
                self.add_updated_var(node_name)
        elif isinstance(node.target, ast.Subscript):
            node_name = node.target.value.id
            if self.is_name_array(node_name):
                self.add_write_trip(
                        node.target,
                        node_name, self.visit(node.target.slice),
                        self.gen_arraydim(node.target))
                self.add_read_trip(
                        node.target,
                        node_name, self.visit(node.target.slice),
                        self.gen_arraydim(node.target))
            else:
                self.add_updated_var(node_name)

        return set([])

    def visit_Call(self, node):
        # catch permute function
        if isinstance(node.func, ast.Name) and node.func.id == "__permute__":
            # permute function has to always have the hypervector type
            # as the first variable, and the values and binary operators 
            # as the second variables
            self.add_read_trip(
                    node,
                    node.args[0].id,
                    self.visit(node.args[1]),
                    FusionIdentifier.gen_permute(node))
            return set([])
        else:
            return self.generic_visit(node)


    def visit_Subscript(self, node):
        # accessed variables
        node_name = node.value.id
        sliceset = self.visit(node.slice)
        if self.is_name_array(node_name):
            self.add_read_trip(
                    node,
                    node_name,
                    sliceset,
                    self.gen_arraydim(node))

        return set(set([node.value.id]) | sliceset)

    def add_read_trip(self, node, name, dependents, replaced_ast):
        #print("READTRIP", name, dependents)
        self.read_trip_vars[self.now_traverse_cNode].append(
                (node, name, dependents, replaced_ast))
        self.add_node_register_name(node, name, dependents, replaced_ast)

    def add_write_trip(self, node, name, dependents, replaced_ast):
        #print("WRITETRIP", name, dependents)
        self.write_trip_vars[self.now_traverse_cNode].append(
                (node, name, dependents, replaced_ast))
        self.add_node_register_name(node, name, dependents, replaced_ast)

    def add_node_register_name(self, node, name, dependents, replaced_ast):
        readable_key = codegen.to_source(replaced_ast)
        dep_str = "_"
        if dependents is not None:
            dep_str += '_'.join([str(d) for d in dependents]) + '_'

        # Take 8 hex digits to avoid large number
        register_name = name + dep_str + \
                str(hashlib.md5(str(readable_key).encode('utf-8')).hexdigest())[:8]

        self.node_to_register_name[node] = register_name
        self.node_to_replaced_ast[node] = replaced_ast
        self.register_name_to_original_name[register_name] = name


    def add_updated_var(self, name):
        self.updated_vars[self.now_traverse_cNode].append(name)

class AccessFusionTransformer(ast.NodeTransformer):
    def __init__(self, fusionIdentifier):
        self.fusionIdentifier = fusionIdentifier

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue

                    if not isinstance(value, ast.For):
                        write_node = self.produce_write(value)
                        if write_node is not None:
                            new_values.extend(write_node)

                    new_values.append(value)

                    if not isinstance(value, ast.For):
                        read_node = self.produce_read(value)
                        if read_node is not None:
                            new_values.extend(read_node)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def produce_write(self, node):
        if node not in self.fusionIdentifier.writes_in_node:
            return None

        ast_writes = [] 
        for _key in self.fusionIdentifier.writes_in_node[node]:
            replace = False
            for var_node in set(
                    self.fusionIdentifier.writes_in_node[node][_key]):
                name = self.fusionIdentifier.node_to_register_name[var_node]
                if var_node not in self.fusionIdentifier.multi_access_nodes:
                    continue
                replace = True

            if replace:
                _, replaced_ast = \
                        self.fusionIdentifier.key_to_replaced_ast[_key]

                if _key in self.fusionIdentifier.shuffled_write:
                    # Overwrite replaced ast for the shuffled write
                    replaced_ast = self.fusionIdentifier.shuffled_write[_key]

                ast_writes.append(
                        ast.Assign(
                            [replaced_ast],
                            ast.Name(name, ast.Load())
                            ))
        return ast_writes

    def produce_read(self, node):
        if node not in self.fusionIdentifier.reads_in_node:
            return None

        ast_reads = [] 
        for _key in self.fusionIdentifier.reads_in_node[node]:
            replace = False
            for var_node in set(
                    self.fusionIdentifier.reads_in_node[node][_key]):
                name = self.fusionIdentifier.node_to_register_name[var_node]
                if var_node not in self.fusionIdentifier.multi_access_nodes:
                    continue
                replace = True

            if replace:
                _, replaced_ast = \
                        self.fusionIdentifier.key_to_replaced_ast[_key]
                ast_reads.append(
                        ast.Assign(
                            [ast.Name(name, ast.Load())],
                            replaced_ast
                            ))
        return ast_reads

    def visit_For(self, node):
        write_node = self.produce_write(node)
        read_node = self.produce_read(node)

        new_node = self.generic_visit(node)

        if read_node is not None:
            new_node.body = read_node + new_node.body

        if write_node is not None:
            new_node.body.extend(write_node)

        return new_node

    def visit_FunctionDef(self, node):
        new_node = self.generic_visit(node)

        read_node = self.produce_read("START")
        write_node = self.produce_write("END")

        if read_node is not None:
            new_node.body = read_node + new_node.body

        if write_node is not None:
            new_node.body.extend(write_node)

        return new_node

class RegisterNameTransformer(ast.NodeTransformer):
    def __init__(self, fusionIdentifier):
        self.fusionIdentifier = fusionIdentifier

    def visit(self, node):
        """Visit a node."""
        if node in self.fusionIdentifier.node_to_register_name:
            if node in self.fusionIdentifier.multi_access_nodes:
                return ast.Name(
                        self.fusionIdentifier.node_to_register_name[node],
                        ast.Load())
            else:
                return self.visit(self.fusionIdentifier.node_to_replaced_ast[node])

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)


