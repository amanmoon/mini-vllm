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
        const ModelConfig &config,
        size_t numTokens,
        int *d_inputTokenIDs,
        void *d_embeddedTokens,
        void *d_embeddingMatrix)
    {
        dim3 grid(numTokens);
        dim3 block(THREADS_PER_BLOCK);

        switch (config.DTYPE)
        {
        case DType::BFloat16:
            embeddingGather<__nv_bfloat16><<<grid, block>>>(
                config.HIDDEN_SIZE,
                d_inputTokenIDs,
                reinterpret_cast<__nv_bfloat16 *>(d_embeddedTokens),
                reinterpret_cast<__nv_bfloat16 *>(d_embeddingMatrix));
            break;

        case DType::Float32:
            embeddingGather<float><<<grid, block>>>(
                config.HIDDEN_SIZE,
                d_inputTokenIDs,
                reinterpret_cast<float *>(d_embeddedTokens),
                reinterpret_cast<float *>(d_embeddingMatrix));
            break;

        default:
            throw std::runtime_error(
                "launchEmbeddingGather: unsupported DType " +
                std::to_string(static_cast<int>(config.DTYPE)));
        }
    }

    void launchRMSNorm(
        const ModelConfig &config,
        size_t numTokens,
        void *d_output,
        void *d_input,
        void *d_normWeights)
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

    void launchCreateRoPETables(
        const ModelConfig &config,
        void *cosTable,
        void *sinTable)
    {
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(config.MAX_POSITION_EMBEDDINGS);

        // RoPE angles are always kept in float for numerical accuracy. This also
        // matches the float accumulator consumed by both supported model dtypes.
        createRoPETables<float><<<grid, block>>>(
            config.MAX_POSITION_EMBEDDINGS,
            config.HEAD_DIM,
            config.ROPE_THETA,
            reinterpret_cast<float *>(cosTable),
            reinterpret_cast<float *>(sinTable));
    }

    void launchRoPEEmbeddings(
        const ModelConfig &config,
        size_t numTokens,
        void *d_inputVector,
        void *d_cosTable,
        void *d_sinTable)
    {
        dim3 grid(numTokens);
        dim3 block(THREADS_PER_BLOCK);

        switch (config.DTYPE)
        {
        case DType::BFloat16:
            RoPEEmbeddings<__nv_bfloat16, float><<<grid, block>>>(
                config.HEAD_DIM,
                reinterpret_cast<__nv_bfloat16 *>(d_inputVector),
                reinterpret_cast<float *>(d_cosTable),
                reinterpret_cast<float *>(d_sinTable));
            break;

        case DType::Float32:
            RoPEEmbeddings<float, float><<<grid, block>>>(
                config.HEAD_DIM,
                reinterpret_cast<float *>(d_inputVector),
                reinterpret_cast<float *>(d_cosTable),
                reinterpret_cast<float *>(d_sinTable));
            break;

        default:
            throw std::runtime_error(
                "launchRoPEEmbeddings: unsupported DType " +
                std::to_string(static_cast<int>(config.DTYPE)));
        }
    }

}
