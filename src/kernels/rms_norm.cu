#include "config.hpp"
#include "kernels.hpp"

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
    __device__ AccT rootMeanSquare(int vectorDim, const T *inputVector, AccT epsilon)
    {
        // allocating more memory than needed, can change to compile time array size.
        __shared__ AccT sharedSum[(THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE];

        AccT localSum = AccT(0);
        const T *in = inputVector + blockIdx.x * vectorDim;
        for (int idx = threadIdx.x; idx < vectorDim; idx += blockDim.x)
        {
            AccT x = static_cast<AccT>(in[idx]);
            localSum += x * x;
        }

        int lane = threadIdx.x % WARP_SIZE;
        int warp = threadIdx.x / WARP_SIZE;
        unsigned warpMask = __activemask();
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            localSum += __shfl_down_sync(warpMask, localSum, offset);

        if (lane == 0)
            sharedSum[warp] = localSum;
        __syncthreads();

        AccT local = AccT(0);

        int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

        unsigned reduceMask = __ballot_sync(0xffffffff, threadIdx.x < numWarps);

        if (threadIdx.x < numWarps)
        {
            local = sharedSum[threadIdx.x];

            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                local += __shfl_down_sync(reduceMask, local, offset);
        }
        if (threadIdx.x == 0)
            sharedSum[0] = sqrtf(local / vectorDim + epsilon);

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
    __global__ void rootMeanSquareNorm(int vectorDim, AccT epsilon, T *outputVector, const T *inputVector, const T *normWeights)
    {
        AccT rmsValue = rootMeanSquare<T, AccT>(vectorDim, inputVector, epsilon);

        AccT invRMSValue = AccT(1) / rmsValue;
        const T *in = inputVector + blockIdx.x * vectorDim;
        T *out = outputVector + blockIdx.x * vectorDim;
        for (int idx = threadIdx.x; idx < vectorDim; idx += blockDim.x)
        {
            AccT normVal = static_cast<AccT>(in[idx]) * invRMSValue * static_cast<AccT>(normWeights[idx]);
            out[idx] = static_cast<T>(normVal);
        }
    }

    template __global__ void rootMeanSquareNorm<__nv_bfloat16, float>(int vectorDim, float epsilon, __nv_bfloat16 *outputVector, const __nv_bfloat16 *inputVector, const __nv_bfloat16 *normWeights);
    template __global__ void rootMeanSquareNorm<float, float>(int vectorDim, float epsilon, float *outputVector, const float *inputVector, const float *normWeights);

    // host, launching function
    void launchRMSNorm(const ModelConfig &config,
                       int numTokens,
                       void *d_output, void *d_input, void *d_normWeights)
    {
        dim3 grid(numTokens);
        dim3 block(THREADS_PER_BLOCK);

        switch (config.DTYPE)
        {
        case DType::BFloat16:
            rootMeanSquareNorm<__nv_bfloat16, float><<<grid, block>>>(
                config.HIDDEN_SIZE,
                config.RMS_NORM_EPS, reinterpret_cast<__nv_bfloat16 *>(d_output),
                reinterpret_cast<__nv_bfloat16 *>(d_input),
                reinterpret_cast<__nv_bfloat16 *>(d_normWeights));
            break;

        case DType::Float32:
            rootMeanSquareNorm<float, double><<<grid, block>>>(
                config.HIDDEN_SIZE,
                static_cast<double>(config.RMS_NORM_EPS),
                reinterpret_cast<float *>(d_output),
                reinterpret_cast<float *>(d_input),
                reinterpret_cast<float *>(d_normWeights));
            break;

        default:
            throw std::runtime_error(
                "launchRMSNorm: unsupported DType " +
                std::to_string(static_cast<int>(config.DTYPE)));
        }
    }
}
