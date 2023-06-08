// =================================
// Math Operators - CUDA part
// =================================
// 
// This implements associtative operations compiled in AOT as runtime library
// (e.g., cos, sign) 

///////////////////////////////////////////
// Defines provided by the AOT compilation
//
// #define __D__ core._D 
//
///////////////////////////////////////////


extern "C" __global__ void cosine(float* hvmatrix) {
    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    const int idx_in_mat = blockIdx.y;
    if (d >= __D__) return;

    hvmatrix[idx_in_mat * __D__ + d] = __cosf(hvmatrix[idx_in_mat * __D__ + d]);
}

extern "C" __global__ void sign(float* hvmatrix) {
    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    const int idx_in_mat = blockIdx.y;
    if (d >= __D__) return;

    const float m = hvmatrix[idx_in_mat * __D__ + d];
    if (m > 0) 
        hvmatrix[idx_in_mat * __D__ + d] = 1.0f;
    else
        hvmatrix[idx_in_mat * __D__ + d] = -1.0f;
}


extern "C" __global__ void fill(float* hvmatrix, float value) {
    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    const int idx_in_mat = blockIdx.y;
    if (d >= __D__) return;

    hvmatrix[idx_in_mat * __D__ + d] = value;
}
