#pragma once

#include "dtype.hpp"

#include <cstddef>

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

}
