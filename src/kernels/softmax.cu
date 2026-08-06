#include "config.hpp"
#include "kernels.cuh"

#include <cuda_bf16.h>

namespace MiniVLLM
{
    /**
     * @brief Computes the row-wise softmax of the attention score matrix.
     *
     * This kernel assumes the attention scores are stored as:
     *
     *      [attentionHead][queryToken][keyToken]
     *
     * where each attention head contains a square matrix of size
     * numTokens × numTokens.
     *
     * Grid Mapping:
     *      grid.x = attentionHeads
     *      grid.y = numTokens (one block per query token)
     *
     * Block Mapping:
     *      block.x = THREADS_PER_BLOCK
     *
     * Each CUDA block computes the softmax for a single row of one
     * attention head. The computation is performed in three stages:
     *
     *  1. Compute the maximum element in the row.
     *     - Each thread scans a subset of columns.
     *     - A parallel reduction finds the row maximum.
     *     - Subtracting this maximum before exponentiation improves
     *       numerical stability.
     *
     *  2. Compute exponentials and their sum.
     *     - Each thread computes exp(score - rowMax) for its assigned
     *       columns.
     *     - The exponentials are written back in-place.
     *     - A parallel reduction computes the sum of all exponentials.
     *
     *  3. Normalize.
     *     - Each stored exponential is divided by the row sum,
     *       producing the final softmax probabilities.
     *
     * Shared memory is reused for both reduction operations.
     *
     * @tparam T Storage type of attention scores
     *           (e.g. float, half, __nv_bfloat16).
     *
     * @param numTokens      Number of tokens in the sequence.
     * @param attentionScore Pointer to the attention score tensor of shape:
     *                       [attentionHeads][numTokens][numTokens].
     */
    template <typename T>
    __global__ void softmax(int numTokens, T *attentionScore)
    {
        __shared__ float shared[THREADS_PER_BLOCK];

        const int head = blockIdx.x;
        const int row = blockIdx.y;

        const int headOffset = head * numTokens * numTokens;
        const int rowOffset = headOffset + row * numTokens;

        float localMax = -HUGE_VALUE_FLOAT;

        for (int col = threadIdx.x; col < numTokens; col += blockDim.x)
        {
            float val = static_cast<float>(attentionScore[rowOffset + col]);
            localMax = fmaxf(localMax, val);
        }

        shared[threadIdx.x] = localMax;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            if (threadIdx.x < offset)
                shared[threadIdx.x] =
                    fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);

            __syncthreads();
        }

        float rowMax = shared[0];
        float localSum = 0.0f;

        for (int col = threadIdx.x; col < numTokens; col += blockDim.x)
        {
            float val = expf(static_cast<float>(attentionScore[rowOffset + col]) - rowMax);

            attentionScore[rowOffset + col] = static_cast<T>(val);
            localSum += val;
        }

        shared[threadIdx.x] = localSum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            if (threadIdx.x < offset)
                shared[threadIdx.x] += shared[threadIdx.x + offset];

            __syncthreads();
        }

        float rowSum = shared[0];

        for (int col = threadIdx.x; col < numTokens; col += blockDim.x)
        {
            attentionScore[rowOffset + col] =
                static_cast<T>(static_cast<float>(attentionScore[rowOffset + col]) / rowSum);
        }
    }

    template __global__ void softmax<__nv_bfloat16>(int numTokens, __nv_bfloat16 *attentionScore);
    template __global__ void softmax<float>(int numTokens, float *attentionScore);
}
