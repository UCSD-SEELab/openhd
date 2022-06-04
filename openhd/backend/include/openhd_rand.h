/*
============================================
OpenHD CUDA Header - random number generator
============================================
*/

// __D__ is defined in the jit-compiled code
#include <curand_kernel.h>

__device__ curandState_t* states[__D__];

extern "C" __global__ void
__launch_bounds__(1024)
__openhd_init_rand_kernel__(int seed) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx < __D__) {
        curandState_t* s = new curandState_t;
        if (s != 0) {
            curand_init(seed, tidx, 0, s);
        }

        states[tidx] = s;
    }
}

__device__ float __draw_random_hypervector__(const int d) {
    curandState_t s = *states[d];
    float val = curand_uniform(&s);
    *states[d] = s;

    //0.2 * 2 = 0.4 => 0
    //0.6 * 2 = 1.2 => 1

    return (int(val * 2) - 1)? -1 : 1;
}

__device__ float __draw_gaussian_hypervector__(const int d) {
    curandState_t s = *states[d];
    float val = curand_normal(&s);
    *states[d] = s;
    return val;
}
