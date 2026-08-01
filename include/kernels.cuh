#pragma once

namespace MiniVLLM
{
    template <typename T>
    __global__ void embeddingGather(int tokenEmbeddingDim, int *inputTokenIDArray, T *embeddedTokenArray, T *embeddingMatrix);

    template <typename T, typename AccT>
    __global__ void rootMeanSquareNorm(int vectorDim, AccT epsilon, T *outputVector, T *inputVector, T *normWeights);

    template <typename AccT>
    __global__ void createRoPETables(int maxSeqLen, int headDim, float base, AccT *cosTable, AccT *sinTable);

    template <typename T, typename AccT>
    __global__ void RoPEEmbeddings(int vectorDim, T *inputVector, AccT *cosTable, AccT *sinTable);

    template <typename T>
    __global__ void residualAdd(int vectorDim, T *inputVector, T *outputVector);
}
