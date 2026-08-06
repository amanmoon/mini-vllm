#include "config.hpp"
#include "kernels.cuh"

#include <cuda_bf16.h>

namespace MiniVLLM
{
    /**
     * @brief Adds a residual connection element-wise.
     *
     * Each CUDA block processes one vector (e.g., one token embedding). Since the
     * vector dimension may exceed the number of threads in a block, each thread
     * iterates over multiple elements with a stride of @c THREADS_PER_BLOCK until
     * the entire vector has been processed.
     *
     * Thread mapping:
     * - Block index (`blockIdx.x`) selects the vector.
     * - Thread index (`threadIdx.x`) selects the starting element within the vector.
     * - Each thread processes:
     *   `threadIdx.x + k * THREADS_PER_BLOCK`
     *   for `k = 0, 1, ...`.
     *
     * @tparam T Data type of the input and output vectors.
     *
     * @param vectorDim   Number of elements in each vector.
     * @param inputVector Input residual vectors of shape `[numVectors, vectorDim]`.
     * @param outputVector Output vectors of shape `[numVectors, vectorDim]`.
     *                     Updated in-place as:
     *                     `outputVector[i] += inputVector[i]`.
     */
    template <typename T>
    __global__ void residualAdd(int vectorDim, T *inputVector, T *outputVector)
    {
        int stride = (vectorDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; ++i)
        {
            int internalBlockDim = threadIdx.x + i * THREADS_PER_BLOCK;
            if (internalBlockDim < vectorDim)
            {
                int workIdx = blockIdx.x * vectorDim + internalBlockDim;
                outputVector[workIdx] += inputVector[workIdx];
            }
        }
    }

    template __global__ void residualAdd<__nv_bfloat16>(int vectorDim, __nv_bfloat16 *inputVector, __nv_bfloat16 *outputVector);
    template __global__ void residualAdd<float>(int vectorDim, float *inputVector, float *outputVector);
}
