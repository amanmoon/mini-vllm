#include "utils.hpp"
#include "load_weights.hpp"
#include "kernels_api.hpp"   // plain-C++ wrappers — no CUDA syntax needed here
#include "llama.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
    const MiniVLLM::ModelConfig &modelConfig = MiniVLLM::Llama::getConfig();

    MiniVLLM::Weights weights;
    MiniVLLM::loadWeightsFromSafeTensor("models/Llama-3.2-1B/model.safetensors", weights);

    std::vector<int> tokens = MiniVLLM::tokenize("Hello, how are you?");
    int numTokens = static_cast<int>(tokens.size());

    int *d_tokens = nullptr;
    cudaMalloc(&d_tokens, tokens.size() * sizeof(int));
    cudaMemcpy(
        d_tokens,
        tokens.data(),
        tokens.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    static constexpr std::size_t BF16_SIZE = 2; 

    void *d_embedded = nullptr;
    cudaMalloc(&d_embedded, numTokens * modelConfig.HIDDEN_SIZE* BF16_SIZE);

    void *embed_weight_ptr =
        static_cast<uint8_t *>(weights.data_ptr) +
        weights.tensors["model.embed_tokens.weight"].dataOffset;

    MiniVLLM::launchEmbeddingGather(
        modelConfig,
        numTokens,
        d_tokens,
        d_embedded,
        embed_weight_ptr);

    cudaDeviceSynchronize();

    std::vector<uint16_t> h_output(numTokens * modelConfig.HIDDEN_SIZE);
    cudaMemcpy(
        h_output.data(),
        d_embedded,
        h_output.size() * BF16_SIZE,
        cudaMemcpyDeviceToHost);

    cudaFree(d_tokens);
    cudaFree(d_embedded);
    cudaDeviceReset();
    return 0;
}
