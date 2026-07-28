
#include <cuda_bf16.h>

namespace MiniVLLM
{
    const int THREADS_PER_BLOCK = 1024;

    // generate embedding vector for each token in the input sequence
    template <typename T>
    __global__ void embeddingGatherKernel(size_t embeddingSize, int *gpuInputTokens, T *inputEmbeddings, T *embedTokens)
    {
        int stride = (embeddingSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        for (int i = 0; i < stride; ++i)
        {
            int embeddingArrayIndex = threadIdx.x + i * THREADS_PER_BLOCK;
            if (embeddingArrayIndex < embeddingSize)
            {
                int outIndex = blockIdx.x * embeddingSize + embeddingArrayIndex;
                int inIndex = gpuInputTokens[blockIdx.x] * embeddingSize + embeddingArrayIndex;
                inputEmbeddings[outIndex] = embedTokens[inIndex];
            }
        }
    }

    template __global__ void embeddingGatherKernel<__nv_bfloat16>(size_t embeddingSize, int *gpuInputTokens, __nv_bfloat16 *inputEmbeddings, __nv_bfloat16 *embedTokens);
}
