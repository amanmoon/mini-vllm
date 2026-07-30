#include "kernels_api.hpp"
#include "kernels.cuh"
#include "config.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace MiniVLLM
{

    void launchEmbeddingGather(
        DType dtype,
        int numTokens,
        size_t tokenEmbeddingDim,
        int *d_inputTokenIDs,
        void *d_embeddedTokens,
        void *d_embeddingMatrix)
    {
        dim3 grid(numTokens);
        dim3 block(THREADS_PER_BLOCK);

        switch (dtype)
        {
        case DType::BFloat16:
            embeddingGatherKernel<__nv_bfloat16><<<grid, block>>>(
                tokenEmbeddingDim,
                d_inputTokenIDs,
                reinterpret_cast<__nv_bfloat16 *>(d_embeddedTokens),
                reinterpret_cast<__nv_bfloat16 *>(d_embeddingMatrix));
            break;

        case DType::Float32:
            embeddingGatherKernel<float><<<grid, block>>>(
                tokenEmbeddingDim,
                d_inputTokenIDs,
                reinterpret_cast<float *>(d_embeddedTokens),
                reinterpret_cast<float *>(d_embeddingMatrix));
            break;

        default:
            throw std::runtime_error(
                "launchEmbeddingGather: unsupported DType " +
                std::to_string(static_cast<int>(dtype)));
        }
    }

    void launchRMSNorm(
        DType dtype,
        int numTokens,
        int vectorDim,
        void *d_output,
        void *d_input,
        void *d_normWeights)
    {
        dim3 grid(numTokens);
        dim3 block(THREADS_PER_BLOCK);

        switch (dtype)
        {
        case DType::BFloat16:
            rootMeanSquareNorm<__nv_bfloat16, float><<<grid, block>>>(
                vectorDim,
                reinterpret_cast<__nv_bfloat16 *>(d_output),
                reinterpret_cast<__nv_bfloat16 *>(d_input),
                reinterpret_cast<__nv_bfloat16 *>(d_normWeights));
            break;

        case DType::Float32:
            rootMeanSquareNorm<float, double><<<grid, block>>>(
                vectorDim,
                reinterpret_cast<float *>(d_output),
                reinterpret_cast<float *>(d_input),
                reinterpret_cast<float *>(d_normWeights));
            break;

        default:
            throw std::runtime_error(
                "launchRMSNorm: unsupported DType " +
                std::to_string(static_cast<int>(dtype)));
        }
    }

}
