#pragma once

#include "dtype.hpp"

#include <cstddef>

namespace MiniVLLM
{
    void launchEmbeddingGather(
        DType dtype,
        int numTokens,
        size_t tokenEmbeddingDim,
        int *d_inputTokenIDs,
        void *d_embeddedTokens,
        void *d_embeddingMatrix);

    void launchRMSNorm(
        DType dtype,
        int numTokens,
        int vectorDim,
        void *d_output,
        void *d_input,
        void *d_normWeights);

}
