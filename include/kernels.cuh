#pragma once

namespace MiniVLLM
{
    template <typename T>
    __global__ void embeddingGatherKernel(size_t tokenEmbeddingDim, int *inputTokenIDArray, T *embeddedTokenArray, T *embeddingMatrix);

    template <typename T, typename AccT>
    __global__ void rootMeanSquareNorm(int vectorDim, T *outputVector, T *inputVector, T *normWeights);
}
