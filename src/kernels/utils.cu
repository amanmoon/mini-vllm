#include "config.hpp"
#include "kernels.cuh"

#include <cuda_bf16.h>

namespace MiniVLLM
{
    /**
     * @brief Applies a causal (lower-triangular) mask to the attention score matrix.
     *
     * Each CUDA block is responsible for masking the attention matrix of one
     * attention head. The attention scores are assumed to be stored as a
     * contiguous tensor of shape:
     *
     *     [attentionHeads][numTokens][numTokens]
     *
     * During causal masking, all positions where the key token appears after
     * the query token (column > row) are assigned a large negative value so
     * that their contribution becomes zero after the softmax operation.
     *
     * Example (numTokens = 4):
     *
     *      0    1    2    3
     *   +---------------------
     * 0 |  ✓    X    X    X
     * 1 |  ✓    ✓    X    X
     * 2 |  ✓    ✓    ✓    X
     * 3 |  ✓    ✓    ✓    ✓
     *
     * ✓ : value is preserved
     * X : replaced with -HUGE_VALUE_FLOAT
     *
     * The work is distributed across the threads in the block by flattening
     * the 2D attention matrix into a 1D array.
     *
     * @tparam T Data type of the attention score matrix
     * @param numTokens Number of tokens in the current sequence.
     * @param attentionScore Pointer to the attention score tensor of shape
     */
    template <typename T>
    __global__ void causalMask(int numTokens, T *attentionScore)
    {
        int total = numTokens * numTokens;
        int stride = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        for (int i = 0; i < stride; i++)
        {
            int internalIdx = threadIdx.x + i * THREADS_PER_BLOCK;

            if (internalIdx < total)
            {
                int row = internalIdx / numTokens;
                int col = internalIdx % numTokens;

                if (col > row)
                {
                    int workIdx = blockIdx.x * total + internalIdx;
                    attentionScore[workIdx] = static_cast<T>(-HUGE_VALUE_FLOAT);
                }
            }
        }
    }

    template __global__ void causalMask<__nv_bfloat16>(int numTokens, __nv_bfloat16 *attentionScore);
    template __global__ void causalMask<float>(int numTokens, float *attentionScore);
}
