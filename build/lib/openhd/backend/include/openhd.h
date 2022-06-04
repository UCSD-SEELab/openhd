/*
=========================
OpenHD CUDA Header - core
=========================
*/

__device__ __forceinline__ float* __hvdim__(
        float* p, const int d) {
    return p + d;
}

__device__ __forceinline__ float* __hxdim__(
        float* p, const int n, const int d) {
    return p + n * __D__ + d;
}

template<typename T>
__device__ __forceinline__ T* __npdim_2d__(
        T* p, const int i1, const int i2, const int stride) {
    return p + i1 * stride + i2;
}


template<typename T>
__device__ __forceinline__ T* __npdim__(
        T* p, const int i) {
    return p + i;
}


__device__ __forceinline__ float* __permute__(
        float* p, const int r, const int d) {
    if (r + d < __D__)
        return p + r + d;
    else
        return p + (r + d - __D__);
}

__device__ __forceinline__ float __flip__(
        float v, const int flipd, const int d) {
    return (d < flipd)? -1 * v : v;
}

__device__ __forceinline__ float __hypervector__(const int d) {
    return 0; // Start computation with 0 element
}


#define __to_int__(x) ((int)(x))
#define __to_float__(x) ((float)(x))
#define __to_bool__(x) ((bool)(x))
