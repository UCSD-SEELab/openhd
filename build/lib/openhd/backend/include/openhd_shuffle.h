/*
=============================
OpenHD CUDA library - shuffle
=============================
*/


// __shuffle_indices__ must be defined with ConstantMemPortioner
__device__ __forceinline__ float* __hv_shuffle__(
        float* p, const int seed, const int d) {
    if (d + seed < __D__)
        return p + __shuffle_indices__[d + seed]; 
    else
        return p + __shuffle_indices__[d + seed - __D__];
}

__device__ __forceinline__ float* __hx_shuffle__(
        float* p, const int n, const int seed, const int d) {
    if (d + seed < __D__)
        return p + n * __D__ + __shuffle_indices__[d + seed]; 
    else
        return p + n * __D__ + __shuffle_indices__[d + seed - __D__];
}
