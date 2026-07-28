#pragma once

namespace MiniVLLM
{
    template <typename T>
    __global__ void embeddingGatherKernel(size_t embeddingSize, int *gpuInputTokens, T *inputEmbeddings, T *embedTokens);
}
