#pragma once

namespace MiniVLLM
{
    template <typename T>
    __global__ void embeddingGather(size_t tokenEmbeddingDim, int *inputTokenIDArray, T *embeddedTokenArray, T *embeddingMatrix);

    template <typename T, typename AccT>
    __global__ void rootMeanSquareNorm(size_t vectorDim, AccT epsilon, T *outputVector, T *inputVector, T *normWeights);

    template <typename AccT>
    __global__ void createRoPETables(size_t maxSeqLen, size_t headDim, float base, AccT *cosTable, AccT *sinTable);

    template <typename T, typename AccT>
    __global__ void RoPEEmbeddings(size_t vectorDim, T *inputVector, AccT *cosTable, AccT *sinTable);
}
