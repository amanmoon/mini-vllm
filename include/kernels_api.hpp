#pragma once

#include "dtype.hpp"

#include <cstddef>
#include <cublas_v2.h>

namespace MiniVLLM
{
    void launchEmbeddingGather(
        const ModelConfig &config,
        int numTokens,
        int *d_inputTokenIDs,
        void *d_embeddedTokens,
        void *d_embeddingMatrix);

    void launchRMSNorm(
        const ModelConfig &config,
        int numTokens,
        void *d_output,
        void *d_input,
        void *d_normWeights);

    void launchCreateRoPETables(
        const ModelConfig &config,
        void *cosTable,
        void *sinTable);

    void launchRoPEEmbeddings(
        const ModelConfig &config,
        int numTokens,
        void *d_inputVector,
        void *d_cosTable,
        void *d_sinTable);

    void launchResidualAdd(
        const ModelConfig &config,
        int numTokens,
        void *d_inputVector,
        void *d_outputVector);

    void launchGEMM(
        const ModelConfig &config,
        int m,
        int n,
        int k,
        void *d_output,
        const void *d_a,
        const void *d_b,
        bool transposeA = false,
        bool transposeB = false,
        float alpha = 1.0f,
        float beta = 0.0f);

    void launchGroupQueryAttention(
        const ModelConfig &config,
        cublasHandle_t cublas_handle,
        int numTokens,
        void *d_qProjection,
        void *d_kProjection,
        void *d_vProjection,
        void *d_oProjection);
}
