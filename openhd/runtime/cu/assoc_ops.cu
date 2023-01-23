// =================================
// Associative Operators - CUDA part
// =================================
// 
// This implements associtative operations compiled in AOT as runtime library
// (e.g., search, cossim) 

///////////////////////////////////////////
// Defines provided by the AOT compilation
//
// #define __D__ core._D 
//
///////////////////////////////////////////


extern "C" __global__ void cossim(
        float* assoc_mem, float* query,
        float* result_array,
        int N_am, int N_query
        )
{
    const int idx_in_query = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_in_query >= N_query)
        return;

    int idx_in_am, d;
    for (idx_in_am = 0; idx_in_am < N_am; ++idx_in_am) {
        float am_sum = 0;
        for (d = 0; d < __D__; ++d) {
            const float m = assoc_mem[idx_in_am * __D__ + d];
            am_sum += m * m;
        }

        float dot = 0;
        float q_sum = 0;
        for (d = 0; d < __D__; ++d) {
            const float m = assoc_mem[idx_in_am * __D__ + d];
            const float q = query[idx_in_query * __D__ + d];
            dot += m * q;
            q_sum += q * q;
        }

        float sim = dot / (sqrt(am_sum) * sqrt(q_sum));
        result_array[idx_in_query * N_am + idx_in_am] = sim;
    }
}

extern "C" __global__ void search(
        float* assoc_mem, float* query,
        int* result_array,
        int N_am, int N_query
        )
{
    const int idx_in_query = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_in_query >= N_query)
        return;

    float maxsim = -1000000.0f; // Small enough number in cosine similarity
    int max_idx_in_am = -1;

    int idx_in_am, d;
    for (idx_in_am = 0; idx_in_am < N_am; ++idx_in_am) {
        float am_sum = 0;
        for (d = 0; d < __D__; ++d) {
            const float m = assoc_mem[idx_in_am * __D__ + d];
            am_sum += m * m;
        }

        float dot = 0;
        float q_sum = 0;
        for (d = 0; d < __D__; ++d) {
            const float m = assoc_mem[idx_in_am * __D__ + d];
            const float q = query[idx_in_query * __D__ + d];
            dot += m * q;
            q_sum += q * q;
        }

        float sim = dot / (sqrt(am_sum) * sqrt(q_sum));

        if (sim > maxsim) {
            maxsim = sim;
            max_idx_in_am = idx_in_am; 
        }
    }

    result_array[idx_in_query] = max_idx_in_am;
}

