#include "load_weights.hpp"
#include "utils.hpp"
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

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
    int numTokens = tokens.size();

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

    __nv_bfloat16 *d_output = nullptr;

    cudaMalloc(
        &d_output,
        numTokens * 2048 * sizeof(__nv_bfloat16));

    __nv_bfloat16 *embed_weight_ptr = reinterpret_cast<__nv_bfloat16 *>(
        static_cast<uint8_t *>(weights.data_ptr) + weights.tensors["model.embed_tokens.weight"].dataOffset);
    MiniVLLM::embeddingGatherKernel<__nv_bfloat16><<<numTokens, 1024>>>(2048, d_tokens, d_output, embed_weight_ptr);

    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_output(numTokens * 2048);

    cudaMemcpy(
        h_output.data(),
        d_output,
        h_output.size() * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << __bfloat162float(h_output[i]) << ' ';

        // Print one embedding per line
        if ((i + 1) % 2048 == 0)
            std::cout << '\n';
    }

    cudaDeviceReset();
    return 0;
}
