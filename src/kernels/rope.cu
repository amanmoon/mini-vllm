#include "config.hpp"
#include "kernels.cuh"

#include <cuda_bf16.h>

namespace MiniVLLM
{
    /**
     * @brief Precomputes the cosine and sine lookup tables used for Rotary Position
     *        Embeddings (RoPE).
     *
     * Generates the rotation values for every `(position, dimension pair)` combination
     * using the RoPE formulation:
     *
     *      θ = position / base^(2 * pair / headDim)
     *
     * The resulting values are stored in row-major order:
     *
     *      table[position][pair]
     *
     * where:
     * - `position ∈ [0, maxSeqLen)`
     * - `pair ∈ [0, headDim / 2)`
     *
     * The tables are later reused during inference to avoid recomputing trigonometric
     * functions for every attention layer.
     *
     * Thread Mapping:
     * - One CUDA block processes one position/token.
     * - Threads within that block compute the position's dimension pairs.
     *
     * Memory Layout:
     * - cosTable[position * (headDim / 2) + pair]
     * - sinTable[position * (headDim / 2) + pair]
     *
     * @tparam AccT  Precision used for the lookup tables (e.g. float,
     *               __nv_bfloat16).
     *
     * @param maxSeqLen Maximum supported sequence length.
     * @param headDim   Dimension of a single attention head. Must be even.
     * @param base      RoPE frequency base (typically 10000).
     * @param cosTable  Output cosine lookup table of size
     *                  maxSeqLen × (headDim / 2).
     * @param sinTable  Output sine lookup table of size
     *                  maxSeqLen × (headDim / 2).
     */
    template <typename AccT>
    __global__ void createRoPETables(int maxSeqLen, int headDim, float base, AccT *cosTable, AccT *sinTable)
    {
        const int position = blockIdx.x;
        if (position >= maxSeqLen)
            return;

        const int numPairs = headDim / 2;
        const int stride = (numPairs + blockDim.x - 1) / blockDim.x;

        for (int i = 0; i < stride; ++i)
        {
            const int pair = threadIdx.x + i * blockDim.x;
            if (pair >= numPairs)
                continue;

            const float exponent =
                (2.0f * static_cast<float>(pair)) / static_cast<float>(headDim);
            const float invFreq = 1.0f / powf(base, exponent);
            const float theta = static_cast<float>(position) * invFreq;
            const int idx = position * numPairs + pair;

            cosTable[idx] = static_cast<AccT>(cosf(theta));
            sinTable[idx] = static_cast<AccT>(sinf(theta));
        }
    }

    template __global__ void createRoPETables<__nv_bfloat16>(int maxSeqLen, int headDim, float base, __nv_bfloat16 *cosTable, __nv_bfloat16 *sinTable);
    template __global__ void createRoPETables<float>(int maxSeqLen, int headDim, float base, float *cosTable, float *sinTable);

    template <typename T, typename AccT>
    __global__ void RoPEEmbeddings(int vectorDim, T *inputVector, AccT *cosTable, AccT *sinTable)
    {
        int stride = ((vectorDim / 2) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        for (int i = 0; i < stride; i++)
        {
            int currPairIdx = threadIdx.x + i * THREADS_PER_BLOCK;
            if (currPairIdx < vectorDim / 2)
            {
                int tableIdx = blockIdx.x * (vectorDim / 2) + currPairIdx;

                AccT cosTheta = cosTable[tableIdx];
                AccT sinTheta = sinTable[tableIdx];

                int workIdx = tableIdx * 2;

                AccT tempIn1 = inputVector[workIdx];
                AccT tempIn2 = inputVector[workIdx + 1];

                inputVector[workIdx] = static_cast<T>(tempIn1 * cosTheta - tempIn2 * sinTheta);
                inputVector[workIdx + 1] = static_cast<T>(tempIn1 * sinTheta + tempIn2 * cosTheta);
            }
        }
    }

    template __global__ void RoPEEmbeddings<__nv_bfloat16, float>(int vectorDim, __nv_bfloat16 *inputVector, float *cosTable, float *sinTable);
    template __global__ void RoPEEmbeddings<float, float>(int vectorDim, float *inputVector, float *cosTable, float *sinTable);
}
