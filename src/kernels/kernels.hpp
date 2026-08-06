#pragma once

#include "utils.hpp"
#include "dtype.hpp"

#include <cublas_v2.h>

namespace MiniVLLM
{
    void launchEmbeddingGather(const ModelConfig &config,
                               int numTokens, int *d_inputTokenIDs,
                               void *d_embeddedTokens, void *d_embeddingMatrix);

    void launchRMSNorm(const ModelConfig &config,
                       int numTokens,
                       void *d_output, void *d_input, void *d_normWeights);
}
