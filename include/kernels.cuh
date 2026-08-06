#pragma once

#include <cublas_v2.h>

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

    template <typename T>
    void getGroupedQueryAttentionScores(int attentionHeads, int keyValueHeads, int headDim, int numTokens, const float &attentionAlpha, const float &attentionBeta,
                                        cublasHandle_t cublas_handle, T *queryProjection, T *keyProjection, T *attentionScore);

    template <typename T>
    __global__ void causalMask(int numTokens, T *attentionScore);

    template <typename T>
    __global__ void softmax(int numTokens, T *attentionScore);

    template <typename T>
    void computeAttentionOutput(int attentionHeads, int keyValueHeads, int headDim, int numTokens, const float &attentionAlpha, const float &attentionBeta,
                                cublasHandle_t cublas_handle, T *attentionScore, T *valueProjection, T *attentionOutput);

    template <typename T>
    void computeOutputProjection(int hiddenSize, int numTokens, const float &alpha, const float &beta,
                                 cublasHandle_t cublasHandle, T *attentionOutput, T *oProjectionWeights, T *output);
}
