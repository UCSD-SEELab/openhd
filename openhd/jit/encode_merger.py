"""
==============
Encoder merger
==============

This merge preprocessor and encode function in a single file
"""

import ast
from ..jit import jit_codegen as codegen

from ..dev import debug
from ..jit import PREPROC_FEATURES, prebuilt_encode_variables

TAG = "jit.encode_merger"

def merge_encode_function(
    str_preprocess_func, preproc_func_vars,
        str_encode_func, encode_func_vars
        ):

    # Create ast
    encode_node = ast.parse(str_encode_func)
    preproc_node = ast.parse(str_preprocess_func)
    input_features_var = encode_func_vars[0]
    output_hypervector_var = encode_func_vars[1]
    original_feature_var = preproc_func_vars[0]
    preprocessed_feature_var = preproc_func_vars[1]


    # Change the variable name
    encode_node = EncoderTransformer(
            input_features_var,
            output_hypervector_var
            ).visit(encode_node)
    preproc_node = PreprocessorTransformer(
            original_feature_var,
            preprocessed_feature_var, 
            input_features_var
            ).convert(preproc_node.body[0].body)


    # Insert the placeholder to the encoding function
    encode_node.body[0].body = [preproc_node] + \
            encode_node.body[0].body

    merged_py_code = codegen.to_source(encode_node)
    debug.log(debug.DEV, TAG, merged_py_code, add_lineno=True)

    return merged_py_code

class PreprocessorTransformer(ast.NodeTransformer):
    def __init__(self,
            original_feature_var, preprocessed_feature_var,
            input_features_var):
        self.original_feature_var = original_feature_var
        self.preprocessed_feature_var = preprocessed_feature_var
        self.input_features_var = input_features_var


    def convert(self, preproc_body):
        TEMPLATE = """
__n__ = __blockIdx_y__
if ((__n__ + __base_n__) >= __N__):
    __hd_return()

if __blockDim_x__ >= __F__ or __threadIdx_x__ < __F__ :
    # number of features processed in each thread
    F_PER_THREAD = (__F__ + __blockDim_x__ - 1) / __blockDim_x__

    sample_idx_in_stream = __stream__ * __M__ * __F__ + __blockIdx_y__ * __F__
    for __f__ in range(F_PER_THREAD):
        __f_idx__ = __threadIdx_x__ * F_PER_THREAD  + __f__
        if __f_idx__ >= __F__:
            break

        %s = %s[sample_idx_in_stream + __f_idx__]
        PREPROC_PLACEHOLDER()
        %s[__f_idx__] = %s

__syncthreads()


__d__ = __threadIdx_x__ + __blockIdx_x__ * __blockDim_x__;
if (__d__ >= __D__):
    __hd_return()
__d__ = __d__ # create barrier for access fusion (will be deleted in py14)
"""
        self.preproc_body = preproc_body

        body = TEMPLATE % (
                self.original_feature_var,
                self.input_features_var,
                PREPROC_FEATURES,
                self.preprocessed_feature_var)
        node = ast.parse(body)
        return self.visit(node)

    def visit_Call(self, node):
        if node.func.id == "PREPROC_PLACEHOLDER":
            return self.preproc_body

        return node

class EncoderTransformer(ast.NodeTransformer):
    def __init__(self, input_features_var, output_hypervector_var):
        self.input_features_var = input_features_var
        self.output_hypervector_var = output_hypervector_var

    def visit_Name(self, node):
        if node.id == self.input_features_var:
            node.id = PREPROC_FEATURES
        if node.id == self.output_hypervector_var:
            return ast.Subscript(
                    ast.Name(node.id, ast.Load()),
                    ast.BinOp(
                        op=ast.Add(),
                        left=ast.Name("__n__", ast.Load()),
                        right=ast.Name("__base_n__", ast.Load()),
                        ctx=node.ctx),
                    node.ctx
                    )


        return node

    def visit_FunctionDef(self, node):
        node.body = [self.visit(c) for c in node.body]
        return node

def get_inputs_wo_encode(inputs, encode_func_vars):
    if encode_func_vars is None:
        return inputs
    return inputs - set(prebuilt_encode_variables) - set([PREPROC_FEATURES]) - \
            set(["__D__", encode_func_vars[1]])

