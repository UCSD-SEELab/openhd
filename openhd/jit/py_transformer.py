"""
==================
Python Transformer
==================

This replaces the planned chunks in the python code to calls for the launchers 
"""
import ast
from ast import iter_fields

from ..jit import jit_codegen as codegen
from ..jit.encode_merger import get_inputs_wo_encode

def replace_python_code(python_node, chunk_list, encode_func_vars):
    """
    Apply PythonTransformer for each chunk
    """
    if len(chunk_list) == 0:
        python_node = PythonTransformer(
                    None, encode_func_vars).visit(python_node)
    else:
        for chunk in chunk_list:
            python_node = PythonTransformer(
                    chunk, encode_func_vars).visit(python_node)

    pycode = codegen.to_source(python_node)

    if encode_func_vars is not None:
        pycode += "\n    return %s\n" % encode_func_vars[1]

    return pycode


class PythonTransformer(ast.NodeTransformer):
    """
    Class to transform a chunk of the original python code
    to a CUDA launching call
    """
    def __init__(self, chunk, encode_func_vars):
        self.chunk = chunk
        if chunk is not None:
            self.first_node = chunk.anode_items[0].item
            self.last_node = chunk.anode_items[-1].item
        else:
            self.first_node = self.last_node = None

        self.encode_func_vars = encode_func_vars
        self.found = False

    call_template = '''RET = FUNC(ARGS)'''
    call_template_no_output = '''FUNC(ARGS)'''
    def replace_to_func_call(self):
        if len(self.chunk.outputs) > 0: 
            body = self.call_template
        else:
            body = self.call_template_no_output

        inputs_wo_encode = get_inputs_wo_encode(
                self.chunk.inputs, self.encode_func_vars)

        body = body.replace('FUNC', self.chunk.get_launcher_func_name())
        body = body.replace('ARGS', ','.join(sorted(list(inputs_wo_encode))))
        body = body.replace('RET', ','.join(sorted(list(self.chunk.outputs))))

        return [ast.parse(body)]

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        if value == self.first_node and field == 'body':
                            self.found = True
                            new_values.extend(self.replace_to_func_call())
                            if value == self.last_node: # first == last
                                self.found = False
                            continue
                        elif self.found:
                            if value == self.last_node:
                                self.found = False
                            continue
                        else:
                            value = self.visit(value)
                            if value is None:
                                continue
                            elif not isinstance(value, ast.AST):
                                new_values.extend(value)
                                continue
                    new_values.append(value)
                old_value[:] = new_values

            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Attribute):
            return node

        if node.func.value.id == "hd": # TODO: Better estimation?
            node.func.value.id = "__hd__"

        return node




