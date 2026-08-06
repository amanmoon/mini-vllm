#include "config.hpp"
#include "kernels.cuh"

#include <cuda_bf16.h>

namespace MiniVLLM
{

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
    __device__ AccT rootMeanSquare(int vectorDim, T *inputVector, AccT epsilon)
    {
        __shared__ AccT sharedSum[THREADS_PER_BLOCK];

        sharedSum[threadIdx.x] = 0.0f;
        int stride = (vectorDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; i++)
        {
            int internalBlockDim = threadIdx.x + i * THREADS_PER_BLOCK;
            if (internalBlockDim < vectorDim)
            {
                int workIdx = blockIdx.x * vectorDim + internalBlockDim;
                sharedSum[threadIdx.x] += static_cast<AccT>(inputVector[workIdx]) * static_cast<AccT>(inputVector[workIdx]);
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
            sharedSum[0] = sqrt((sharedSum[0] / static_cast<AccT>(vectorDim)) + epsilon);

        __syncthreads();

        return sharedSum[0];
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
     *     RMS(input) = sqrt((1 / vectorDim) * Σ(input_i²) + epsilon)
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
    __global__ void rootMeanSquareNorm(int vectorDim, AccT epsilon, T *outputVector, T *inputVector, T *normWeights)
    {
        AccT rmsValue = rootMeanSquare<T, AccT>(vectorDim, inputVector, epsilon);

        int stride = (vectorDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; ++i)
        {
            int internalBlockDim = i * THREADS_PER_BLOCK + threadIdx.x;
            if (internalBlockDim < vectorDim)
            {
                int workerIdx = blockIdx.x * vectorDim + internalBlockDim;
                AccT out =
                    static_cast<AccT>(inputVector[workerIdx]) /
                    rmsValue *
                    static_cast<AccT>(normWeights[internalBlockDim]);

                outputVector[workerIdx] = static_cast<T>(out);
            }
        }
    }

    template __global__ void rootMeanSquareNorm<__nv_bfloat16, float>(int vectorDim, float epsilon, __nv_bfloat16 *outputVector, __nv_bfloat16 *inputVector, __nv_bfloat16 *normWeights);
    template __global__ void rootMeanSquareNorm<float, double>(int vectorDim, double epsilon, float *outputVector, float *inputVector, float *normWeights);
}
