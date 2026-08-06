#include "kernels.cuh"
#include "llama.hpp"
#include "dtype.hpp"
#include "config.hpp"

#include <cuda_bf16.h>

namespace MiniVLLM
{
    /**
     * @brief Generate embedding vector for each token in the input sequence.
     *
     * Each CUDA block processes one input token. Threads within the block
     * cooperatively copy the corresponding embedding vector from the embedding
     * matrix into the output buffer.
     *
     * @tparam T Data type of the embedding values.
     *
     * @param hiddenSize Size of each embedding vector (hidden dimension).
     * @param inputTokenArray Device pointer to an array of input token IDs of
     *        length `numTokens`. Block `i` processes `inputTokenIDArray[i]`.
     * @param outputEmbeddedArray Device pointer to the output buffer of shape
     *        `[numTokens, tokenEmbeddingDim]` where the gathered embeddings are stored.
     * @param embeddingMatrix Device pointer to the embedding lookup table of shape
     *        `[vocabSize, tokenEmbeddingDim]`. Each row corresponds to the embedding
     *        vector of a token ID.
     */

    template <typename T>
    __global__ void embeddingGather(int hiddenSize, const int *inputTokenArray, T *outputEmbeddedArray, const T *embeddingMatrix)
    {
        int tokenID = inputTokenArray[blockIdx.x];

        T *out = outputEmbeddedArray + blockIdx.x * hiddenSize;
        const T *in = embeddingMatrix + tokenID * hiddenSize;

        for (int idx = threadIdx.x; idx < hiddenSize; idx += blockDim.x)
        {
            out[idx] = in[idx];
        }
    }

    template __global__ void embeddingGather<__nv_bfloat16>(int hiddenSize, const int *inputTokenArray, __nv_bfloat16 *outputEmbeddedArray, const __nv_bfloat16 *embeddingMatrix);
    template __global__ void embeddingGather<float>(int hiddenSize, const int *inputTokenArray, float *outputEmbeddedArray, const float *embeddingMatrix);

    void launchEmbeddingGather(
        const ModelConfig &config,
        int numTokens,
        int *d_inputTokenIDs,
        void *d_embeddedTokens,
        void *d_embeddingMatrix)
    {
        dim3 grid(numTokens);
        dim3 block(THREADS_PER_BLOCK);

        switch (config.DTYPE)
        {
        case DType::BFloat16:
            embeddingGather<__nv_bfloat16><<<grid, block>>>(
                config.HIDDEN_SIZE,
                d_inputTokenIDs,
                reinterpret_cast<__nv_bfloat16 *>(d_embeddedTokens),
                reinterpret_cast<__nv_bfloat16 *>(d_embeddingMatrix));
            break;

        case DType::Float32:
            embeddingGather<float><<<grid, block>>>(
                config.HIDDEN_SIZE,
                d_inputTokenIDs,
                reinterpret_cast<float *>(d_embeddedTokens),
                reinterpret_cast<float *>(d_embeddingMatrix));
            break;

        default:
            throw std::runtime_error(
                "launchEmbeddingGather: unsupported DType " +
                std::to_string(static_cast<int>(config.DTYPE)));
        }
    }
}
