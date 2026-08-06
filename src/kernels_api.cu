#include "kernels_api.hpp"
#include "kernels.cuh"
#include "config.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace MiniVLLM
{

    void launchEmbeddingGather(
        const ModelConfig &config,
        int numTokens,
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
        int numTokens,
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

        createRoPETables<float><<<grid, block>>>(
            config.MAX_POSITION_EMBEDDINGS,
            config.HEAD_DIM,
            config.ROPE_THETA,
            reinterpret_cast<float *>(cosTable),
            reinterpret_cast<float *>(sinTable));
    }

    void launchRoPEEmbeddings(
        const ModelConfig &config,
        int numTokens,
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

    void launchResidualAdd(
        const ModelConfig &config,
        int numTokens,
        void *d_inputVector,
        void *d_outputVector)
    {
        dim3 grid(numTokens);
        dim3 block(THREADS_PER_BLOCK);

        switch (config.DTYPE)
        {
        case DType::BFloat16:
            residualAdd<__nv_bfloat16><<<grid, block>>>(
                config.HIDDEN_SIZE,
                reinterpret_cast<__nv_bfloat16 *>(d_inputVector),
                reinterpret_cast<__nv_bfloat16 *>(d_outputVector));
            break;

        case DType::Float32:
            residualAdd<float><<<grid, block>>>(
                config.HIDDEN_SIZE,
                reinterpret_cast<float *>(d_inputVector),
                reinterpret_cast<float *>(d_outputVector));
            break;

        default:
            throw std::runtime_error(
                "launchResidualAdd: unsupported DType " +
                std::to_string(static_cast<int>(config.DTYPE)));
        }
    }

    void launchGEMM(
        const ModelConfig &config,
        cublasHandle_t cublas_handle,
        int m, int n, int k,
        void *d_output, const void *d_a, const void *d_b,
        bool transposeA, bool transposeB,
        float alpha, float beta)
    {
        cudaDataType_t dataType;
        if (config.DTYPE == DType::BFloat16)
            dataType = CUDA_R_16BF;
        else if (config.DTYPE == DType::Float32)
            dataType = CUDA_R_32F;
        else
            throw std::runtime_error(
                "launchGEMM: unsupported DType " +
                std::to_string(static_cast<int>(config.DTYPE)));

        const cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

        const int lda = transposeA ? m : k;
        const int ldb = transposeB ? k : n;

        const cublasStatus_t gemmStatus = cublasGemmEx(
            cublas_handle, opB, opA,
            n, m, k,
            &alpha, d_b, dataType, ldb,
            d_a, dataType, lda, &beta,
            d_output, dataType, n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        if (gemmStatus != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(
                "cublasGemmEx failed with cuBLAS status " +
                std::to_string(static_cast<int>(gemmStatus)));
        }
    }

    template <typename T>
    void attention(cublasHandle_t cublas_handle,
                   int attentionHeads,
                   int keyValueHeads,
                   int headDim,
                   int numTokens,
                   T *queryProjection,
                   T *keyProjection,
                   T *valueProjection,
                   T *oProjectionWeight,
                   T *attentionScore,
                   T *attentionOutput,
                   T *output)
    {
        float keyQueryAlpha = 1.0f / sqrt(headDim);
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        getGroupedQueryAttentionScores<T>(
            attentionHeads,
            keyValueHeads,
            headDim,
            numTokens,
            keyQueryAlpha,
            beta,
            cublas_handle,
            queryProjection,
            keyProjection,
            attentionScore);

        causalMask<<<attentionHeads, THREADS_PER_BLOCK>>>(
            numTokens,
            attentionScore);

        dim3 softmaxGrid(attentionHeads, numTokens);

        softmax<T><<<softmaxGrid, THREADS_PER_BLOCK>>>(
            numTokens,
            attentionScore);

        computeAttentionOutput<T>(
            attentionHeads,
            keyValueHeads,
            headDim,
            numTokens,
            alpha,
            beta,
            cublas_handle,
            attentionScore,
            valueProjection,
            attentionOutput);

        computeOutputProjection<T>(
            attentionHeads * headDim,
            numTokens,
            alpha,
            beta,
            cublas_handle,
            attentionOutput,
            oProjectionWeight,
            output);
    }

    void launchGroupQueryAttention(
        const ModelConfig &config,
        cublasHandle_t cublas_handle,
        int numTokens,
        void *d_qProjection,
        void *d_kProjection,
        void *d_vProjection,
        void *d_oProjectionWeight,
        void *d_output)
    {
        const size_t attentionScoreSize =
            static_cast<size_t>(config.NUM_ATTENTION_HEADS) *
            numTokens * numTokens;

        const size_t attentionContextSize =
            static_cast<size_t>(config.HIDDEN_SIZE) *
            numTokens;

        switch (config.DTYPE)
        {
        case DType::BFloat16:
        {
            __nv_bfloat16 *d_attentionScore;
            __nv_bfloat16 *d_attentionContext;

            cudaMalloc(&d_attentionScore,
                       attentionScoreSize * sizeof(__nv_bfloat16));

            cudaMalloc(&d_attentionContext,
                       attentionContextSize * sizeof(__nv_bfloat16));

            attention<__nv_bfloat16>(
                cublas_handle,
                config.NUM_ATTENTION_HEADS,
                config.NUM_KEY_VALUE_HEADS,
                config.HEAD_DIM,
                numTokens,
                reinterpret_cast<__nv_bfloat16 *>(d_qProjection),
                reinterpret_cast<__nv_bfloat16 *>(d_kProjection),
                reinterpret_cast<__nv_bfloat16 *>(d_vProjection),
                reinterpret_cast<__nv_bfloat16 *>(d_oProjectionWeight),
                d_attentionScore,
                d_attentionContext,
                reinterpret_cast<__nv_bfloat16 *>(d_output));

            cudaFree(d_attentionScore);
            cudaFree(d_attentionContext);
            break;
        }

        case DType::Float32:
        {
            float *d_attentionScore;
            float *d_attentionContext;

            cudaMalloc(&d_attentionScore,
                       attentionScoreSize * sizeof(float));

            cudaMalloc(&d_attentionContext,
                       attentionContextSize * sizeof(float));

            attention<float>(
                cublas_handle,
                config.NUM_ATTENTION_HEADS,
                config.NUM_KEY_VALUE_HEADS,
                config.HEAD_DIM,
                numTokens,
                reinterpret_cast<float *>(d_qProjection),
                reinterpret_cast<float *>(d_kProjection),
                reinterpret_cast<float *>(d_vProjection),
                reinterpret_cast<float *>(d_oProjectionWeight),
                d_attentionScore,
                d_attentionContext,
                reinterpret_cast<float *>(d_output));

            cudaFree(d_attentionScore);
            cudaFree(d_attentionContext);
            break;
        }

        default:
            throw std::runtime_error(
                "launchGroupQueryAttention: unsupported DType " +
                std::to_string(static_cast<int>(config.DTYPE)));
        }
    }
}
