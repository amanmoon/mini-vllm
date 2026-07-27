
#include <iostream>
#include <filesystem>

// CUDA
#include <cuda_runtime.h>

namespace fs = std::filesystem;

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

void validateSafeTensorFile(const std::string& filename)
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