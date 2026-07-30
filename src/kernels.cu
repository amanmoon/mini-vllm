#include "config.hpp"

#include <cuda_bf16.h>

namespace MiniVLLM
{

    /**
     * @brief Generate embedding vector for each token in the input sequence.
     *
     * Each CUDA block processes one input token. Threads within the block
     * cooperatively copy the corresponding embedding vector from the embedding
     * matrix into the output buffer.
     *
     * @tparam T Data type of the embedding values (e.g. __nv_bfloat16, float).
     *
     * @param tokenEmbeddingDim Size of each embedding vector (hidden dimension).
     * @param inputTokenIDArray Device pointer to an array of input token IDs of
     *        length `numTokens`. Block `i` processes `inputTokenIDArray[i]`.
     * @param embeddedTokenArray Device pointer to the output buffer of shape
     *        `[numTokens, tokenEmbeddingDim]` where the gathered embeddings are stored.
     * @param embeddingMatrix Device pointer to the embedding lookup table of shape
     *        `[vocabSize, tokenEmbeddingDim]`. Each row corresponds to the embedding
     *        vector of a token ID.
     */

    template <typename T>
    __global__ void embeddingGatherKernel(size_t tokenEmbeddingDim, int *inputTokenIDArray, T *embeddedTokenArray, T *embeddingMatrix)
    {
        int stride = (tokenEmbeddingDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; ++i)
        {
            int embeddingArrayIndex = threadIdx.x + i * THREADS_PER_BLOCK;
            if (embeddingArrayIndex < tokenEmbeddingDim)
            {
                int outIndex = blockIdx.x * tokenEmbeddingDim + embeddingArrayIndex;
                int inIndex = inputTokenIDArray[blockIdx.x] * tokenEmbeddingDim + embeddingArrayIndex;
                embeddedTokenArray[outIndex] = embeddingMatrix[inIndex];
            }
        }
    }

    template __global__ void embeddingGatherKernel<__nv_bfloat16>(size_t tokenEmbeddingDim, int *inputTokenIDArray, __nv_bfloat16 *embeddedTokenArray, __nv_bfloat16 *embeddingMatrix);
    template __global__ void embeddingGatherKernel<float>(size_t tokenEmbeddingDim, int *inputTokenIDArray, float *embeddedTokenArray, float *embeddingMatrix);

    /**
     * @brief Computes the Root Mean Square (RMS) of a vector.
     *
     * The RMS is computed as:
     *
     *     sqrt((1 / vectorDim) * Σ(x_i²) + epsilon)
     *
     * @tparam T Input data type (e.g. float, __half, __nv_bfloat16).
     * @tparam AccT is the Data type of accumulator used for increased numerical stability.
     *
     * @param vectorDim Length of the input vector.
     * @param inputVector Pointer to the input vectors stored in contiguous memory.
     * @param epsilon For preventing zero division.
     *
     * @return The RMS value of the vector assigned to the current CUDA block.
     *
     * @note
     * - One CUDA block processes one input vector.
     * - Accumulation is performed in float precision for improved numerical
     *   stability.
     * - The returned RMS value is identical for every thread in the block.
     */

    template <typename T, typename AccT>
    __device__ T rootMeanSquare(int vectorDim, T *inputVector, float epsilon)
    {
        __shared__ float sharedSum[THREADS_PER_BLOCK];

        sharedSum[threadIdx.x] = 0.0f;
        int stride = (vectorDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; i++)
        {
            int internalBlockDim = threadIdx.x + i * THREADS_PER_BLOCK;
            if (internalBlockDim < vectorDim)
            {
                int workIdx = blockIdx.x * vectorDim + internalBlockDim;
                sharedSum[threadIdx.x] += (AccT)inputVector[workIdx] * (AccT)inputVector[workIdx];
            }
        }
        __syncthreads();
        for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                sharedSum[threadIdx.x] += sharedSum[threadIdx.x + stride];

            __syncthreads();
        }
        if (threadIdx.x == 0)
            sharedSum[0] = sqrtf((sharedSum[0] / vectorDim) + epsilon);

        __syncthreads();

        return (T)sharedSum[0];
    }

    /**
     * @brief Applies Root Mean Square Normalization (RMSNorm) to one or more vectors.
     *
     * Each CUDA block normalizes one token vector. The kernel first computes the
     * vector's RMS using `rootMeanSquare()`, then divides every element by the RMS
     * and applies a learnable per-dimension scaling factor.
     *
     * For each element:
     *
     *     output_i = (input_i / RMS(input)) * normWeight_i
     *
     * where
     *
     *     RMS(input) = sqrt((1 / vectorDim) * Σ(input_i²) + RMS_NORM_EPS)
     *
     * @tparam T Data type of the input, output, and normalization weights.
     * @tparam AccT is the Data type of accumulator in the rootMeanSquare function used for increased numerical stability.
     *
     * @param vectorDim Dimension of each token vector.
     * @param outputVector Pointer to the output vectors.
     * @param inputVector Pointer to the input vectors.
     * @param normWeights Pointer to the learnable RMSNorm scale weights.
     *
     * @note
     * - Grid layout: one CUDA block per token vector.
     * - Computation of the RMS internally uses AccT accumulation for improved
     *   numerical stability.
     */
    template <typename T, typename AccT>
    __global__ void rootMeanSquareNorm(int vectorDim, T *outputVector, T *inputVector, T *normWeights)
    {
        T rmsValue = rootMeanSquare<T, AccT>(vectorDim, inputVector, RMS_NORM_EPS);

        int stride = (vectorDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; ++i)
        {
            int internalBlockDim = i * THREADS_PER_BLOCK + threadIdx.x;
            if (internalBlockDim < vectorDim)
            {
                int workerIdx = blockIdx.x * vectorDim + internalBlockDim;
                outputVector[workerIdx] = inputVector[workerIdx] / rmsValue * normWeights[workerIdx];
            }
        }
    }

    template __global__ void rootMeanSquareNorm<__nv_bfloat16, float>(int vectorDim, __nv_bfloat16 *outputVector, __nv_bfloat16 *inputVector, __nv_bfloat16 *normWeights);
    template __global__ void rootMeanSquareNorm<float, double>(int vectorDim, float *outputVector, float *inputVector, float *normWeights);
}
