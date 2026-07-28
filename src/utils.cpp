
#include <iostream>
#include <filesystem>
#include <array>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// CUDA
#include <cuda_runtime.h>

namespace fs = std::filesystem;

namespace MiniVLLM
{

    void checkGpuAvailablity()
    {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess)
        {
            std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (deviceCount == 0)
        {
            std::cout << "ERROR: No CUDA-capable devices found." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        else
        {
            std::cout << "Number of CUDA-capable devices: " << deviceCount << std::endl;
        }
    }

    void validateSafeTensorFileExists(const std::string &filename)
    {
        if (!fs::exists(filename))
        {
            std::cerr << "Error: File '" << filename << "' does not exist.\n";
            std::exit(EXIT_FAILURE);
        }

        if (fs::path(filename).extension() != ".safetensors")
        {
            std::cerr << "Error: Expected a .safetensors file.\n";
            std::exit(EXIT_FAILURE);
        }
    }

    // run python tokenizer script to tokenize the input text and return the tokens as a vector of integers
    std::vector<int> tokenize(const std::string &text)
    {
        std::string command =
            "python3 python/tokenizer.py \"" + text + "\"";

        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
        {
            throw std::runtime_error("Failed to start tokenizer.");
        }

        std::array<char, 256> buffer;
        std::string output;

        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
        {
            output += buffer.data();
        }

        pclose(pipe);

        std::vector<int> tokens;
        std::stringstream ss(output);

        int token;
        while (ss >> token)
        {
            tokens.push_back(token);
        }

        return tokens;
    }
}
