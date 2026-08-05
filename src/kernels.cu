#include "config.hpp"

#include <cuda_bf16.h>
#include <cmath>
#include <cublas_v2.h>

namespace MiniVLLM
{

    /**
     * @brief Generate embedding vector for each token in the input sequence.
     *
     * Each CUDA block processes one input token. Threads within the block
     * cooperatively copy the corresponding embedding vector from the embedding
     * matrix into the output buffer.
     *
     * @tparam T Data type of the embedding values (e.g. __nv_bfloat16, float).
     *
     * @param tokenEmbeddingDim Size of each embedding vector (hidden dimension).
     * @param inputTokenIDArray Device pointer to an array of input token IDs of
     *        length `numTokens`. Block `i` processes `inputTokenIDArray[i]`.
     * @param embeddedTokenArray Device pointer to the output buffer of shape
     *        `[numTokens, tokenEmbeddingDim]` where the gathered embeddings are stored.
     * @param embeddingMatrix Device pointer to the embedding lookup table of shape
     *        `[vocabSize, tokenEmbeddingDim]`. Each row corresponds to the embedding
     *        vector of a token ID.
     */

    template <typename T>
    __global__ void embeddingGather(int tokenEmbeddingDim, int *inputTokenIDArray, T *embeddedTokenArray, T *embeddingMatrix)
    {
        int stride = (tokenEmbeddingDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; ++i)
        {
            int embeddingArrayIndex = threadIdx.x + i * THREADS_PER_BLOCK;
            if (embeddingArrayIndex < tokenEmbeddingDim)
            {
                int outIndex = blockIdx.x * tokenEmbeddingDim + embeddingArrayIndex;
                int inIndex = inputTokenIDArray[blockIdx.x] * tokenEmbeddingDim + embeddingArrayIndex;
                embeddedTokenArray[outIndex] = embeddingMatrix[inIndex];
            }
        }
    }

    template __global__ void embeddingGather<__nv_bfloat16>(int tokenEmbeddingDim, int *inputTokenIDArray, __nv_bfloat16 *embeddedTokenArray, __nv_bfloat16 *embeddingMatrix);
    template __global__ void embeddingGather<float>(int tokenEmbeddingDim, int *inputTokenIDArray, float *embeddedTokenArray, float *embeddingMatrix);

    /**
     * @brief Computes the Root Mean Square (RMS) of a vector.
     *
     * The RMS is computed as:
     *
     *     sqrt((1 / vectorDim) * Σ(x_i²) + epsilon)
     *
     * @tparam T Input data type (e.g. float, __half, __nv_bfloat16).
     * @tparam AccT is the Data type of accumulator used for increased numerical stability.
     *
     * @param vectorDim Length of the input vector.
     * @param inputVector Pointer to the input vectors stored in contiguous memory.
     * @param epsilon For preventing zero division.
     *
     * @return The RMS value of the vector assigned to the current CUDA block.
     *
     * @note
     * - One CUDA block processes one input vector.
     * - Accumulation is performed in float precision for improved numerical
     *   stability.
     * - The returned RMS value is identical for every thread in the block.
     */

    template <typename T, typename AccT>
    __device__ AccT rootMeanSquare(int vectorDim, T *inputVector, AccT epsilon)
    {
        __shared__ AccT sharedSum[THREADS_PER_BLOCK];

        sharedSum[threadIdx.x] = 0.0f;
        int stride = (vectorDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; i++)
        {
            int internalBlockDim = threadIdx.x + i * THREADS_PER_BLOCK;
            if (internalBlockDim < vectorDim)
            {
                int workIdx = blockIdx.x * vectorDim + internalBlockDim;
                sharedSum[threadIdx.x] += static_cast<AccT>(inputVector[workIdx]) * static_cast<AccT>(inputVector[workIdx]);
            }
        }
        __syncthreads();
        for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                sharedSum[threadIdx.x] += sharedSum[threadIdx.x + stride];

            __syncthreads();
        }
        if (threadIdx.x == 0)
            sharedSum[0] = sqrt((sharedSum[0] / static_cast<AccT>(vectorDim)) + epsilon);

        __syncthreads();

        return sharedSum[0];
    }

    /**
     * @brief Applies Root Mean Square Normalization (RMSNorm) to one or more vectors.
     *
     * Each CUDA block normalizes one token vector. The kernel first computes the
     * vector's RMS using `rootMeanSquare()`, then divides every element by the RMS
     * and applies a learnable per-dimension scaling factor.
     *
     * For each element:
     *
     *     output_i = (input_i / RMS(input)) * normWeight_i
     *
     * where
     *
     *     RMS(input) = sqrt((1 / vectorDim) * Σ(input_i²) + epsilon)
     *
     * @tparam T Data type of the input, output, and normalization weights.
     * @tparam AccT is the Data type of accumulator in the rootMeanSquare function used for increased numerical stability.
     *
     * @param vectorDim Dimension of each token vector.
     * @param outputVector Pointer to the output vectors.
     * @param inputVector Pointer to the input vectors.
     * @param normWeights Pointer to the learnable RMSNorm scale weights.
     *
     * @note
     * - Grid layout: one CUDA block per token vector.
     * - Computation of the RMS internally uses AccT accumulation for improved
     *   numerical stability.
     */
    template <typename T, typename AccT>
    __global__ void rootMeanSquareNorm(int vectorDim, AccT epsilon, T *outputVector, T *inputVector, T *normWeights)
    {
        AccT rmsValue = rootMeanSquare<T, AccT>(vectorDim, inputVector, epsilon);

        int stride = (vectorDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; ++i)
        {
            int internalBlockDim = i * THREADS_PER_BLOCK + threadIdx.x;
            if (internalBlockDim < vectorDim)
            {
                int workerIdx = blockIdx.x * vectorDim + internalBlockDim;
                AccT out =
                    static_cast<AccT>(inputVector[workerIdx]) /
                    rmsValue *
                    static_cast<AccT>(normWeights[internalBlockDim]);

                outputVector[workerIdx] = static_cast<T>(out);
            }
        }
    }

    template __global__ void rootMeanSquareNorm<__nv_bfloat16, float>(int vectorDim, float epsilon, __nv_bfloat16 *outputVector, __nv_bfloat16 *inputVector, __nv_bfloat16 *normWeights);
    template __global__ void rootMeanSquareNorm<float, double>(int vectorDim, double epsilon, float *outputVector, float *inputVector, float *normWeights);

    /**
     * @brief Precomputes the cosine and sine lookup tables used for Rotary Position
     *        Embeddings (RoPE).
     *
     * Generates the rotation values for every `(position, dimension pair)` combination
     * using the RoPE formulation:
     *
     *      θ = position / base^(2 * pair / headDim)
     *
     * The resulting values are stored in row-major order:
     *
     *      table[position][pair]
     *
     * where:
     * - `position ∈ [0, maxSeqLen)`
     * - `pair ∈ [0, headDim / 2)`
     *
     * The tables are later reused during inference to avoid recomputing trigonometric
     * functions for every attention layer.
     *
     * Thread Mapping:
     * - One CUDA block processes one position/token.
     * - Threads within that block compute the position's dimension pairs.
     *
     * Memory Layout:
     * - cosTable[position * (headDim / 2) + pair]
     * - sinTable[position * (headDim / 2) + pair]
     *
     * @tparam AccT  Precision used for the lookup tables (e.g. float,
     *               __nv_bfloat16).
     *
     * @param maxSeqLen Maximum supported sequence length.
     * @param headDim   Dimension of a single attention head. Must be even.
     * @param base      RoPE frequency base (typically 10000).
     * @param cosTable  Output cosine lookup table of size
     *                  maxSeqLen × (headDim / 2).
     * @param sinTable  Output sine lookup table of size
     *                  maxSeqLen × (headDim / 2).
     */
    template <typename AccT>
    __global__ void createRoPETables(int maxSeqLen, int headDim, float base, AccT *cosTable, AccT *sinTable)
    {
        const int position = blockIdx.x;
        if (position >= maxSeqLen)
            return;

        const int numPairs = headDim / 2;
        const int stride = (numPairs + blockDim.x - 1) / blockDim.x;

        for (int i = 0; i < stride; ++i)
        {
            const int pair = threadIdx.x + i * blockDim.x;
            if (pair >= numPairs)
                continue;

            const float exponent =
                (2.0f * static_cast<float>(pair)) / static_cast<float>(headDim);
            const float invFreq = 1.0f / powf(base, exponent);
            const float theta = static_cast<float>(position) * invFreq;
            const int idx = position * numPairs + pair;

            cosTable[idx] = static_cast<AccT>(cosf(theta));
            sinTable[idx] = static_cast<AccT>(sinf(theta));
        }
    }

    template __global__ void createRoPETables<__nv_bfloat16>(int maxSeqLen, int headDim, float base, __nv_bfloat16 *cosTable, __nv_bfloat16 *sinTable);
    template __global__ void createRoPETables<float>(int maxSeqLen, int headDim, float base, float *cosTable, float *sinTable);

    template <typename T, typename AccT>
    __global__ void RoPEEmbeddings(int vectorDim, T *inputVector, AccT *cosTable, AccT *sinTable)
    {
        int stride = ((vectorDim / 2) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        for (int i = 0; i < stride; i++)
        {
            int currPairIdx = threadIdx.x + i * THREADS_PER_BLOCK;
            if (currPairIdx < vectorDim / 2)
            {
                int tableIdx = blockIdx.x * (vectorDim / 2) + currPairIdx;

                AccT cosTheta = cosTable[tableIdx];
                AccT sinTheta = sinTable[tableIdx];

                int workIdx = tableIdx * 2;

                AccT tempIn1 = inputVector[workIdx];
                AccT tempIn2 = inputVector[workIdx + 1];

                inputVector[workIdx] = static_cast<T>(tempIn1 * cosTheta - tempIn2 * sinTheta);
                inputVector[workIdx + 1] = static_cast<T>(tempIn1 * sinTheta + tempIn2 * cosTheta);
            }
        }
    }

    template __global__ void RoPEEmbeddings<__nv_bfloat16, float>(int vectorDim, __nv_bfloat16 *inputVector, float *cosTable, float *sinTable);
    template __global__ void RoPEEmbeddings<float, float>(int vectorDim, float *inputVector, float *cosTable, float *sinTable);

    /**
     * @brief Adds a residual connection element-wise.
     *
     * Each CUDA block processes one vector (e.g., one token embedding). Since the
     * vector dimension may exceed the number of threads in a block, each thread
     * iterates over multiple elements with a stride of @c THREADS_PER_BLOCK until
     * the entire vector has been processed.
     *
     * Thread mapping:
     * - Block index (`blockIdx.x`) selects the vector.
     * - Thread index (`threadIdx.x`) selects the starting element within the vector.
     * - Each thread processes:
     *   `threadIdx.x + k * THREADS_PER_BLOCK`
     *   for `k = 0, 1, ...`.
     *
     * @tparam T Data type of the input and output vectors.
     *
     * @param vectorDim   Number of elements in each vector.
     * @param inputVector Input residual vectors of shape `[numVectors, vectorDim]`.
     * @param outputVector Output vectors of shape `[numVectors, vectorDim]`.
     *                     Updated in-place as:
     *                     `outputVector[i] += inputVector[i]`.
     */
    template <typename T>
    __global__ void residualAdd(int vectorDim, T *inputVector, T *outputVector)
    {
        int stride = (vectorDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; ++i)
        {
            int internalBlockDim = threadIdx.x + i * THREADS_PER_BLOCK;
            if (internalBlockDim < vectorDim)
            {
                int workIdx = blockIdx.x * vectorDim + internalBlockDim;
                outputVector[workIdx] += inputVector[workIdx];
            }
        }
    }
    template __global__ void residualAdd<__nv_bfloat16>(int vectorDim, __nv_bfloat16 *inputVector, __nv_bfloat16 *outputVector);
    template __global__ void residualAdd<float>(int vectorDim, float *inputVector, float *outputVector);

    /**
     * @brief Computes scaled dot-product attention scores for all query heads using
     *        Grouped Query Attention (GQA).
     *
     * Each query head computes:
     *
     *      AttentionScores = Q_head × K_headᵀ
     *
     * where:
     *  - Q_head has shape (numTokens, headDim)
     *  - K_head has shape (numTokens, headDim)
     *  - AttentionScores has shape (numTokens, numTokens)
     *
     * In GQA, multiple query heads share the same key head. The mapping is:
     *
     *      keyValueHead = queryHead / (attentionHeads / keyValueHeads)
     *
     * For example, with 32 query heads and 8 key/value heads:
     *
     *      Q0-Q3   -> KV0
     *      Q4-Q7   -> KV1
     *      ...
     *      Q28-Q31 -> KV7
     *
     * The function computes the attention score matrix for every query head
     * independently using cuBLAS GEMM.
     *
     * @tparam T Data type of the projections (e.g. __nv_bfloat16).
     *
     * @param attentionHeads Number of query attention heads.
     * @param keyValueHeads Number of key/value heads.
     * @param headDim Dimension of each attention head.
     * @param numTokens Number of tokens in the sequence.
     * @param attentionAlpha GEMM scaling factor α.
     * @param attentionBeta GEMM accumulation factor β.
     * @param queryProjection Pointer to the query projection tensor with layout
     *        (numTokens, attentionHeads * headDim).
     * @param keyProjection Pointer to the key projection tensor with layout
     *        (numTokens, keyValueHeads * headDim).
     * @param attentionScore Output buffer storing attention score matrices with
     *        layout (attentionHeads, numTokens, numTokens).
     *
     * @note The projection tensors are assumed to be stored contiguously in
     *       token-major order, where each token contains all attention head
     *       vectors sequentially.
     *
     * @note Accumulation is performed in FP32 while the input and output tensors
     *       are stored in type T.
     */
    template <typename T>
    __device__ void getGroupedQueryAttentionScores(int attentionHeads, int keyValueHeads, int headDim, int numTokens, const float &attentionAlpha, const float &attentionBeta,
                                                   cublasHandle_t cublas_handle, T *queryProjection, T *keyProjection, T *attentionScore)
    {
        int gqaRatio = attentionHeads / keyValueHeads;
        for (int headNum = 0; headNum < attentionHeads; headNum++)
        {
            int keyValueIdx = headNum / gqaRatio;

            T *queryHead = queryProjection + headNum * headDim;
            T *keyHead = keyProjection + keyValueIdx * headDim;
            T *attentionHead = attentionScore + headNum * numTokens * numTokens;

            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_T,
                                                            CUBLAS_OP_N,
                                                            numTokens,
                                                            numTokens,
                                                            headDim,
                                                            &attentionAlpha,
                                                            keyHead,
                                                            CUDA_R_16BF,
                                                            keyValueHeads * headDim,
                                                            queryHead,
                                                            CUDA_R_16BF,
                                                            attentionHeads * headDim,
                                                            &attentionBeta,
                                                            attentionHead,
                                                            CUDA_R_16BF,
                                                            numTokens,
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }
    }

    /**
     * @brief Applies a causal (lower-triangular) mask to the attention score matrix.
     *
     * Each CUDA block is responsible for masking the attention matrix of one
     * attention head. The attention scores are assumed to be stored as a
     * contiguous tensor of shape:
     *
     *     [attentionHeads][numTokens][numTokens]
     *
     * During causal masking, all positions where the key token appears after
     * the query token (column > row) are assigned a large negative value so
     * that their contribution becomes zero after the softmax operation.
     *
     * Example (numTokens = 4):
     *
     *      0    1    2    3
     *   +---------------------
     * 0 |  ✓    X    X    X
     * 1 |  ✓    ✓    X    X
     * 2 |  ✓    ✓    ✓    X
     * 3 |  ✓    ✓    ✓    ✓
     *
     * ✓ : value is preserved
     * X : replaced with -HUGE_VALUE_FLOAT
     *
     * The work is distributed across the threads in the block by flattening
     * the 2D attention matrix into a 1D array.
     *
     * @tparam T Data type of the attention score matrix
     * @param numTokens Number of tokens in the current sequence.
     * @param attentionScore Pointer to the attention score tensor of shape
     */
    template <typename T>
    __device__ void causalMask(int numTokens, T *attentionScore)
    {
        int total = numTokens * numTokens;
        int stride = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        for (int i = 0; i < stride; i++)
        {
            int internalIdx = threadIdx.x + i * THREADS_PER_BLOCK;

            if (internalIdx < total)
            {
                int row = internalIdx / numTokens;
                int col = internalIdx % numTokens;

                if (col > row)
                {
                    int workIdx = blockIdx.x * total + internalIdx;
                    attentionScore[workIdx] = static_cast<T>(-HUGE_VALUE_FLOAT);
                }
            }
        }
    }

    template <typename T>
    __device__ void softmax(int numTokens, T *attentionScore)
    {
        __shared__ float shared[THREADS_PER_BLOCK];

        const int head = blockIdx.x;
        const int row = blockIdx.y;

        const int headOffset = head * numTokens * numTokens;
        const int rowOffset = headOffset + row * numTokens;

        float localMax = -HUGE_VALUE_FLOAT;

        for (int col = threadIdx.x; col < numTokens; col += blockDim.x)
        {
            float val = static_cast<float>(attentionScore[rowOffset + col]);
            localMax = fmaxf(localMax, val);
        }

        shared[threadIdx.x] = localMax;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            if (threadIdx.x < offset)
                shared[threadIdx.x] =
                    fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);

            __syncthreads();
        }

        float rowMax = shared[0];
        float localSum = 0.0f;

        for (int col = threadIdx.x; col < numTokens; col += blockDim.x)
        {
            float val = expf(static_cast<float>(attentionScore[rowOffset + col]) - rowMax);

            attentionScore[rowOffset + col] = static_cast<T>(val);
            localSum += val;
        }

        shared[threadIdx.x] = localSum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
        {
            if (threadIdx.x < offset)
                shared[threadIdx.x] += shared[threadIdx.x + offset];

            __syncthreads();
        }

        float rowSum = shared[0];

        for (int col = threadIdx.x; col < numTokens; col += blockDim.x)
        {
            attentionScore[rowOffset + col] =
                static_cast<T>(static_cast<float>(attentionScore[rowOffset + col]) / rowSum);
        }
    }

    template <typename T>
    __device__ void attentionOutput(int attentionHeads, int keyValueHeads, int headDim, int numTokens, const float &attentionAlpha, const float &attentionBeta,
                                    cublasHandle_t cublas_handle, T *attentionScore, T *valueProjection, T *attentionOutput)
    {
        int gqaRatio = attentionHeads / keyValueHeads;
        for (int headNum = 0; headNum < attentionHeads; headNum++)
        {
            int keyValueIdx = headNum / gqaRatio;

            T *valueHead = valueProjection + keyValueIdx * headDim;
            T *attentionScoreHead = attentionScore + headNum * numTokens * numTokens;
            T *outputHead = attentionOutput + headNum * headDim;

            cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                            CUBLAS_OP_N,
                                                            CUBLAS_OP_N,
                                                            headDim,
                                                            numTokens,
                                                            numTokens,
                                                            &alpha,
                                                            valueHead,
                                                            CUDA_R_16BF,
                                                            keyValueHeads * headDim,
                                                            attentionScoreHead,
                                                            CUDA_R_16BF,
                                                            numTokens,
                                                            &beta,
                                                            outputHead,
                                                            CUDA_R_16BF,
                                                            attentionHeads * headDim,
                                                            CUBLAS_COMPUTE_32F,
                                                            CUBLAS_GEMM_DEFAULT);
        }
    }
}
