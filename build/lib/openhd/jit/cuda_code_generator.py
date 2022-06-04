"""
==============
CUDA Generator
==============

This transpiles two types of code from the extracted chunk:
i) cuda code (CUDA code on GPU)
ii) cuda launching code (python on CPU)
"""
import ast
import hashlib
import pycuda
import numpy as np

from ..dev import debug

from .. import core

from ..jit import jit_codegen as codegen
from ..jit import HYPERVECTOR_TYPE, HYPERMATRIX_TYPE, ARG_PREFIX, STRIDE_POSTFIX
from ..jit import NP_INT_ARRAY_TYPE, NP_FLOAT_ARRAY_TYPE
from ..jit import PREPROC_FEATURES, FEATURE_STREAM, SHUFFLE_INDICES
from ..jit import prebuilt_encode_variables, cuda_builtin_vars
from ..jit import TRANSPILED_KERNEL_FUNCTION
from ..jit.encode_merger import get_inputs_wo_encode

from ..memory.mem_portioner import ConstantMemPortioner

from .permute_unnester import unnest_permute
from .shuffle_transformer import convert_shuffle
#from .assoc_op_transformer import convert_assoc_op
from .access_fusion_transformer import fuse_access
from .data_type_mutator import mutate_data_type
from .hd_name_transformer import map_hd_names

from ..py14 import transpiler
from ..backend.cuda_impl import load_module_from_code_str

TAG = "jit.cuda_code_generator"

def generate_pycuda_code(chunk_list, arg_variables, encode_function_vars):
    """
    Convert each chunk to CUDA code with launchers
    """
    code = ''
    cubin_list = []
    for chunk in chunk_list:
        launcher_code, cubin_filename = transpile_cuda_chunk(
                chunk, arg_variables, encode_function_vars)
        code += launcher_code + '\n\n'
        cubin_list.append(cubin_filename)
    #assert(False)

    return code, cubin_list

def transpile_cuda_chunk(chunk, arg_variables, encode_function_vars):
    """
    Transpile a chunk, compile the cubin, and return the launcher
    """
    # Create cuda code
    cuda_code, used_func_names, arg_variables = CudaCodeGenerator(
            chunk, arg_variables, encode_function_vars).generate()
    debug.log(debug.DEV, TAG, cuda_code, add_lineno=True)

    # Load cubin module by compiling code string
    cubin_filename = load_module_from_code_str(cuda_code) 

    # Generate the launcher
    if encode_function_vars is None:
        launcher_generator = LauncherGenerator(
                chunk, cubin_filename, used_func_names, encode_function_vars)
    else:
        launcher_generator = EncodingLauncherGenerator(
                chunk, cubin_filename, used_func_names,
                arg_variables, encode_function_vars)
    launcher_code = launcher_generator.generate()

    #debug.log(debug.DEV, TAG, launcher_code, add_lineno=True)

    return launcher_code, cubin_filename

class CudaCodeGenerator(object):
    def __init__(self, chunk, arg_variables, encode_function_vars):
        self.chunk = chunk
        self.arg_variables = arg_variables
        self.encode_function_vars = encode_function_vars

    def generate(self):
        """
        Generate code running on CUDA side based on py14 module
        """
        body = '''def TRANSPILED_KERNEL_FUNCTION(ARGS):
    pass
'''
        # 0. Create ast tree to run py14
        cinput = process_cross_platform_inputs(
                self.chunk, self.encode_function_vars)
        debug.log(debug.DEV, TAG, "C-INPUTS: " + str(cinput))
        debug.log(debug.DEV, TAG, "VARTYPES: " + str(self.chunk.used_var_types))
        debug.log(debug.DEV, TAG, "INPUTS: " + str(self.chunk.inputs))
        debug.log(debug.DEV, TAG, "OUTPUTS: " + str(self.chunk.outputs))
        debug.log(debug.DEV, TAG, "CONSTS: " + str(self.chunk.consts))

        body = body.replace(
                'TRANSPILED_KERNEL_FUNCTION', TRANSPILED_KERNEL_FUNCTION)
        body = body.replace('ARGS', ','.join(
            ARG_PREFIX + var \
                    if ARG_PREFIX + var in self.chunk.used_var_types \
                    else var for var in cinput))

        node = ast.parse(body)
        #print(ast.dump(node))
        node.body[0].body = [c.item for c in self.chunk.anode_items]
        #print(ast.dump(node))

        # 1. Preprocessing - transform the ast to the CUDA-understandable ast  
        # Note: After this step, ast addresses in the annotation nodes
        # are invalidated, since we build another ast
        node = unnest_permute(node, self.chunk.used_var_types)
        #print(ast.dump(node))
        node = convert_shuffle(node)
        #print(ast.dump(node))
        #node = convert_assoc_op(node)
        #print(ast.dump(node))

        # Access fusion
        node = fuse_access(node, self.chunk)

        # Data type mutation
        node = mutate_data_type(node, self.chunk, self.arg_variables)

        node, self.used_func_names = map_hd_names(
                node, self.encode_function_vars)
        #print(ast.dump(node))
        self.chunk.anode_items = None # Just make sure that it's not used again
        print(self.chunk.used_var_types)


        # 2. Create c compatiable code
        node = ast.fix_missing_locations(node)
        debug.log(debug.DEV, TAG, codegen.to_source(node), add_lineno=True)
        cuda_code = transpiler.transpile_cuda_from_ast(
                node, self.chunk.used_var_types)
        #print(cuda_code)
        #assert(False)

        # 3. Post processing: add CUDA-specific code
        cuda_code = insert_newline(
                cuda_code, 1,
                '\n' + self.create_function_init_code() + '\n')
        cuda_code = insert_newline(
                cuda_code, -2, '\n' + self.create_function_exit_code() + '\n')

        # Insert defines and headers
        cuda_code = self.defines() + '\n' + \
                self.headers() + '\n' + cuda_code

        return cuda_code, self.used_func_names, self.arg_variables 

    def defines(self):
        # Constant memory definition
        const_body = ""
        const_portioner = ConstantMemPortioner()
        if "__hv_shuffle__" in self.used_func_names or \
                "__hx_shuffle__" in self.used_func_names:
            const_body += const_portioner.allocate(
                    SHUFFLE_INDICES, "__D__", {"__D__": core._D},
                    core.__SHUFFLE_INDICES_CPU__.dtype)

        if self.encode_function_vars is not None:
            # Decide the streaming size
            avail_mem = const_portioner.get_available_mem_size()
            __N_STREAM__ = core.__N_STREAM__
            __F__ = self.arg_variables["__F__"][1]
            __M__ = int(avail_mem / (__F__ * 4) / core.__N_STREAM__) 
            self.arg_variables["__M__"] = (int, __M__) # Update

            # Add code
            const_body += "// Encoding stream will interleave this variable\n"
            const_body += const_portioner.allocate(
                    FEATURE_STREAM,
                    "__N_STREAM__ * __M__ * __F__", {
                        "__N_STREAM__": __N_STREAM__,
                        "__M__": __M__,
                        "__F__": __F__,
                        },
                    np.float32)

        # Macro definition
        TEMPLATE = "#define %s %s\n"
        macro_body = TEMPLATE % ('__D__', str(core._D))

        for var in self.chunk.consts:
            if var in cuda_builtin_vars:
                continue
            _, val = self.arg_variables[var]
            macro_body += TEMPLATE % (var, str(val)) 

        if self.encode_function_vars is not None:
            macro_body += TEMPLATE % ('__N_STREAM__', str(__N_STREAM__))
                        
        return macro_body + "\n" + const_body

    def headers(self):
        TEMPLATE = "#include \"%s\"\n"

        header_files = ["openhd.h"]

        if "__hv_shuffle__" in self.used_func_names or \
                "__hx_shuffle__" in self.used_func_names:
            header_files.append("openhd_shuffle.h")

        if "__draw_random_hypervector__" in self.used_func_names or \
                "__draw_gaussian_hypervector__" in self.used_func_names:
            header_files.append("openhd_rand.h")
        
        body = ''
        for filename in header_files:
            body += TEMPLATE % filename

        return body

    def create_function_init_code(self):
        init_code = ""
        if PREPROC_FEATURES in self.chunk.inputs:
            init_code += self.add_shared_encode_mem()

        init_code += self.declare_cuda_block()
        init_code += self.args_to_func_var_code()

        return init_code

    def declare_cuda_block(self):
        TEMPLATE = """const int __d__ = threadIdx.x + blockIdx.x * blockDim.x;
if (__d__ >= __D__) return;
"""
        if PREPROC_FEATURES in self.chunk.inputs:
            return "" # Encoding function already has it in the encode merger

        return TEMPLATE


    def add_shared_encode_mem(self):
        TEMPLATE = "__shared__ float %s%s[__F__];\n"
        return TEMPLATE % (ARG_PREFIX, PREPROC_FEATURES)


    def args_to_func_var_code(self):
        TEMPLATE = "%s = *%s;\n"

        body = ""
        for var in self.chunk.inputs:
            var_type = self.chunk.used_var_types[var]
            if var_type == HYPERVECTOR_TYPE or \
                    var_type == HYPERMATRIX_TYPE or \
                    var_type == NP_FLOAT_ARRAY_TYPE or \
                    var_type == NP_INT_ARRAY_TYPE:
                continue # Array types are handled by fused access 

            arg_var = ARG_PREFIX + var 
            if arg_var in self.chunk.used_var_types:
                body += TEMPLATE % (var, arg_var)

        return body

    def create_function_exit_code(self):
        exit_code = self.func_var_to_args_code()
        return exit_code

    def func_var_to_args_code(self):
        """ First thread copy all the data """
        # TODO: __n__
        TEMPLATE_IF = "if (__d__ == 0) {\n"
        TEMPLATE = "*%s = %s;\n"
        TEMPLATE_FI = "}\n"

        body = ""
        for var in self.chunk.outputs:
            var_type = self.chunk.used_var_types[var]
            if var_type == HYPERVECTOR_TYPE or \
                    var_type == HYPERMATRIX_TYPE or \
                    var_type == NP_FLOAT_ARRAY_TYPE or \
                    var_type == NP_INT_ARRAY_TYPE:
                continue # Array types are handled by fused access 


            arg_var = ARG_PREFIX + var 
            if arg_var in self.chunk.used_var_types:
                body += TEMPLATE % (arg_var, var)
            else:
                # Output has to have the address
                assert(False)

        if len(body) > 0:
            body = TEMPLATE_IF + body + TEMPLATE_FI

        return body

class LauncherGenerator(object):
    def __init__(
            self, chunk, cubin_filename, used_func_names, encode_function_vars):
        self.chunk = chunk
        self.cubin_filename = cubin_filename
        self.used_func_names = used_func_names
        self.encode_function_vars = encode_function_vars

    def generate(self):
        # Function header
        inputs_wo_encode = get_inputs_wo_encode(
                self.chunk.inputs, self.encode_function_vars)

        body = 'def %s(%s):\n' % (
                self.chunk.get_launcher_func_name(),
                ', '.join(sorted(list(inputs_wo_encode))))

        # Load cubin
        body += tab(self.launcher_load_cubin())

        # Allocation
        body += tab(self.launcher_hd_allocs())

        # Call cuda function
        body += tab(self.launcher_call_func())

        # Function return
        if len(self.chunk.outputs) > 0:
            body += tab('return (%s)\n' % (','.join(sorted(list(
                self.chunk.outputs)))))

        return body

    def launcher_load_cubin(self):
        TEMPLATE = """
# Load modules and run cuda initailizer if needed
__MAX_THREADS__ = __ci__.DeviceData().max_threads

mod__ = __ci__.get_module(\"FILENAME\")
cuda_func__ = mod__.get_function(\"TRANSPILED_KERNEL_FUNCTION\")
"""

        body = TEMPLATE
        body = body.replace("FILENAME", self.cubin_filename)
        body = body.replace(
                'TRANSPILED_KERNEL_FUNCTION', TRANSPILED_KERNEL_FUNCTION)

        if "__hv_shuffle__" in self.used_func_names or \
                "__hx_shuffle__" in self.used_func_names:
            body += """
const_shuffle_indices_mem_addr = mod__.get_global(\"%s\")[0]
__ci__.drv.memcpy_htod(const_shuffle_indices_mem_addr, __SHUFFLE_INDICES_CPU__)
""" % SHUFFLE_INDICES

        if "__draw_random_hypervector__" in self.used_func_names or \
                "__draw_gaussian_hypervector__" in self.used_func_names:
            body += """
__openhd_init_rand_kernel__ = mod__.get_function(\"__openhd_init_rand_kernel__\")
__openhd_init_rand_kernel__(__np__.int32(__time__.time()),
    block=(__MAX_THREADS__, 1, 1),
    grid=((__D__ + __MAX_THREADS__ - 1) // __MAX_THREADS__, 1))
"""
        return body


    def launcher_hd_allocs(self):
        TEMPLATE = "%s = __hd__.hypervector()\n"
        if len(self.chunk.allocs) == 0:
            return ""

        body = '# Allocate new GPU memories for Hypervector or Hypermatrix\n'
        for name in self.chunk.allocs:
            body += TEMPLATE % name

        return body


    def launcher_call_func(self):
        TEMPLATE = """
cuda_func__(
    INPUTS,
    block=(__MAX_THREADS__, 1, 1),
    grid=((__D__ + __MAX_THREADS__ - 1) // __MAX_THREADS__, 1))

"""
        body = ""

        # Get the cross-platform inputs
        cinput = process_cross_platform_inputs(
                self.chunk, self.encode_function_vars)
        pointer_type_inputs = self.get_pointer_type_inputs(cinput)

        # Add the Host-GPU communication if any
        if len(pointer_type_inputs) > 0:
            body += "__arr__ = __HD_ARG_ARRAY__\n"
            body += "__arr__.reset()\n"
            need_push = False
            for var in pointer_type_inputs:
                var_type = self.chunk.used_var_types[var]
                if not isinstance(var_type, str):
                    var_type = var_type.__name__

                if var not in self.chunk.inputs:
                    body += "__arr__.add" + \
                            ("(0, \"%s\")\n" % (var_type))
                else:
                    body += "__arr__.add" + \
                            ("(%s, \"%s\")\n" % (var, var_type))
                    need_push = True

            #if need_push:
            body += "__arr__.push()\n"

        # Copy numpy array to GPU
        for var in cinput:
            var_type = self.chunk.used_var_types[var]
            if var_type == NP_INT_ARRAY_TYPE or var_type == NP_FLOAT_ARRAY_TYPE:
                if var in self.chunk.hd_arrays:
                    # HD array
                    body += "%s, %s = %s.get_gpu_mem_structure()\n" % (
                            var, ARG_PREFIX + var + STRIDE_POSTFIX, var)
                else:
                    # Numpy array
                    body += "%s, %s = __ci__.declare_cuda_array(%s)\n" % (
                            var, ARG_PREFIX + var + STRIDE_POSTFIX, var)
            

        # Create the cuda call 
        body += TEMPLATE.replace('INPUTS', self.create_input_str(
            cinput, pointer_type_inputs))

        # Add the write-back code if any
        if len(pointer_type_inputs) > 0:
            body += "__arr__.pull()\n"
            for idx, var in enumerate(pointer_type_inputs):
                var_type = self.chunk.used_var_types[var]
                if not isinstance(var_type, str):
                    var_type = var_type.__name__

                body += "%s = __arr__[%d][0]\n" % (var, idx)

        return body

    def get_pointer_type_inputs(self, cinput):
        pointer_type_inputs = []
        for var in cinput:
            var_type = self.chunk.used_var_types[var]
            if var_type == HYPERVECTOR_TYPE or var_type == HYPERMATRIX_TYPE:
                continue

            if var_type == NP_INT_ARRAY_TYPE or var_type == NP_FLOAT_ARRAY_TYPE:
                continue

            if ARG_PREFIX + var in self.chunk.used_var_types:
                pointer_type_inputs.append(var)

        return pointer_type_inputs

    def create_input_str(self, cinput, pointer_type_inputs):
        input_argstr_list = []
        for var in cinput:
            var_type = self.chunk.used_var_types[var]
            if var in pointer_type_inputs:
                idx = pointer_type_inputs.index(var)
                input_argstr_list.append("__arr__.addr(%d)" % idx)
            elif var_type == HYPERVECTOR_TYPE or \
                    var_type == HYPERMATRIX_TYPE:
                input_argstr_list.append(var + '.mem')
            else:
                if var_type == 'int' or var_type == int:
                    TYPE_TEMPLATE = "__np__.int32(%s)"
                elif var_type == 'float' or var_type == float:
                    TYPE_TEMPLATE = "__np__.float32(%s)"
                elif var_type == 'bool' or var_type == bool:
                    TYPE_TEMPLATE = "__np__.bool(%s)"
                else:
                    TYPE_TEMPLATE = "%s"

                input_argstr_list.append(TYPE_TEMPLATE % var)

        return ', '.join(input_argstr_list)


class EncodingLauncherGenerator(LauncherGenerator):
    """ Inherent class to generate encoding launcher """
    def __init__(
            self, chunk, cubin_filename, used_func_names,
            arg_variables, encode_function_vars):
        self.arg_variables = arg_variables
        super(EncodingLauncherGenerator, self).__init__(
                chunk, cubin_filename, used_func_names,
                encode_function_vars)

    def launcher_hd_allocs(self):
        debug.thrifty_assert(
                len(self.chunk.allocs) == 0,
                "Encoding function does not allow internal "
                "hypervector allocation(s).")

        # Create Hypermatrix to store the encoded value
        body = "%s = __hd__.hypermatrix(%s.shape[0])\n" % (
                self.encode_function_vars[1], self.encode_function_vars[0])

        return body

    def launcher_call_func(self):
        TEMPLATE = """
__M__ = N_STREAMED_SAMPLES
const_feature_mem_addr = mod__.get_global(\"%s\")[0]
streamer = __ci__.Streamer(FEATURE_NAME, const_feature_mem_addr, __M__)
for (__base_n__, __stream__, __drv_stream__) in streamer:
    cuda_func__(
        INPUTS,
        block=(__MAX_THREADS__, 1, 1),
        grid=((__D__ + __MAX_THREADS__ - 1) // __MAX_THREADS__, __M__),
        stream=__drv_stream__)

""" % FEATURE_STREAM
        body = ""

        # Get the cross-platform inputs
        cinput = process_cross_platform_inputs(
                self.chunk, self.encode_function_vars)

        pointer_type_inputs = self.get_pointer_type_inputs(cinput)
        debug.thrifty_assert(
                len(pointer_type_inputs) == 0,
                "Encoding function does not allow any return variables.")

        # Copy numpy array to GPU
        for var in cinput:
            if var in self.encode_function_vars[0]:
                continue

            var_type = self.chunk.used_var_types[var]
            if var_type == NP_INT_ARRAY_TYPE or var_type == NP_FLOAT_ARRAY_TYPE:
                body += "%s, %s = __ci__.declare_cuda_array(%s)\n" % (
                        var, ARG_PREFIX + var + STRIDE_POSTFIX, var)
                debug.log(
                        debug.WARNING, TAG,
                        "Your encoding function accesses numpy array on GPU. " +
                        "It may slow down the process."
                        )
            

        # Create the cuda call 
        call_body = TEMPLATE.replace(
                'N_STREAMED_SAMPLES', str(self.arg_variables["__M__"][1]))
        call_body = call_body.replace(
                'FEATURE_NAME', self.encode_function_vars[0])
        call_body = call_body.replace(
                'INPUTS', self.create_input_str(cinput, pointer_type_inputs))
        body += call_body
                        
        return body


def process_cross_platform_inputs(chunk, encode_function_vars):
    # Determine input variables
    all_vars = set(chunk.inputs) | set(chunk.allocs) | set(chunk.outputs)
    all_vars -= set(chunk.consts)

    if encode_function_vars is not None:
        # Generated variables in CUDA Code generator
        all_vars -= set([PREPROC_FEATURES, encode_function_vars[0]])

    # Give the type for input variables
    args_var_types = dict()
    extra_strides = []
    for var in all_vars:
        var_type = chunk.used_var_types[var]
        if not isinstance(var_type, str):
            var_type = var_type.__name__

        if var_type == HYPERVECTOR_TYPE or var_type == HYPERMATRIX_TYPE:
            # The type will be actually determined in the data type mutation
            args_var_types[ARG_PREFIX + var] = var_type
            continue

        if var_type == NP_INT_ARRAY_TYPE:
            args_var_types[ARG_PREFIX + var] = "int*"
            args_var_types[ARG_PREFIX + var + STRIDE_POSTFIX] = "const int"
            extra_strides.append(ARG_PREFIX + var + STRIDE_POSTFIX) 
            continue

        if var_type == NP_FLOAT_ARRAY_TYPE:
            args_var_types[ARG_PREFIX + var] = "float*"
            args_var_types[ARG_PREFIX + var + STRIDE_POSTFIX] = "const int"
            extra_strides.append(ARG_PREFIX + var + STRIDE_POSTFIX) 
            continue

        if var in chunk.outputs:
            args_var_types[ARG_PREFIX + var] = var_type + "*"
        else:
            # If it's not output variable, we don't change the name 
            # for readability
            args_var_types[var] = var_type

    chunk.used_var_types.update(args_var_types)
    all_vars |= set(extra_strides)

    return sorted(list(all_vars))

def tab(body):
    TAB = '    '
    return '\n'.join(TAB + line for line in body.split('\n')) + '\n'

def insert_newline(code, index, line):
    code_lines = code.split('\n')
    code_lines.insert(index, line)
    return '\n'.join(code_lines) + '\n'

