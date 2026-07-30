#include "utils.hpp"
#include "load_weights.hpp"
#include "kernels_api.hpp"   // plain-C++ wrappers — no CUDA syntax needed here

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
    MiniVLLM::Weights weights;
    MiniVLLM::loadWeightsFromSafeTensor("models/Llama-3.2-1B/model.safetensors", weights);
    // for (const auto &[key, tensor] : weights.tensors)
    // {
    //     std::cout << "Tensor: " << key << std::endl;
    //     std::cout << "Data Type: " << tensor.dataType << std::endl;
    //     std::cout << "Shape: ";
    //     for (const auto &dim : tensor.shape)
    //     {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "Data Offset: " << tensor.dataOffset << std::endl;
    //     std::cout << "Data Size: " << tensor.dataSize << std::endl;
    //     std::cout << "------------------------" << std::endl;
    // }

    std::vector<int> tokens = MiniVLLM::tokenize("Hello, how are you?");
    int numTokens = static_cast<int>(tokens.size());

    // for (int token : tokens)
    // {
    //     std::cout << token << ' ';
    // }

    int *d_tokens = nullptr;
    cudaMalloc(&d_tokens, tokens.size() * sizeof(int));
    cudaMemcpy(
        d_tokens,
        tokens.data(),
        tokens.size() * sizeof(int),
        cudaMemcpyHostToDevice);

    static constexpr int EMBED_DIM = 2048;
    static constexpr std::size_t BF16_SIZE = 2; 

    void *d_embedded = nullptr;
    cudaMalloc(&d_embedded, numTokens * EMBED_DIM * BF16_SIZE);

    void *embed_weight_ptr =
        static_cast<uint8_t *>(weights.data_ptr) +
        weights.tensors["model.embed_tokens.weight"].dataOffset;

    MiniVLLM::launchEmbeddingGather(
        MiniVLLM::DType::BFloat16,
        numTokens,
        EMBED_DIM,
        d_tokens,
        d_embedded,
        embed_weight_ptr);

    cudaDeviceSynchronize();

    std::vector<uint16_t> h_output(numTokens * EMBED_DIM);
    cudaMemcpy(
        h_output.data(),
        d_embedded,
        h_output.size() * BF16_SIZE,
        cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10; ++i)
    {
        uint32_t bits = static_cast<uint32_t>(h_output[i]) << 16;
        float val;
        __builtin_memcpy(&val, &bits, sizeof(float));
        std::cout << val << ' ';

        if ((i + 1) % EMBED_DIM == 0)
            std::cout << '\n';
    }

    cudaFree(d_tokens);
    cudaFree(d_embedded);
    cudaDeviceReset();
    return 0;
}

