// rapidjson/example/simpledom/simpledom.cpp`
#include "rapidjson/document.h"

// internal headers
#include "utils.hpp"
#include "dtype.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <fstream>

// CUDA
#include <cuda_runtime.h>

namespace MiniVLLM
{

    void loadWeightsFromSafeTensor(const std::string &modelFilePath, Weights &weights)
    {

        validateSafeTensorFileExists(modelFilePath);
        std::ifstream safeTensorFile(modelFilePath, std::ios::binary);

        // read the first 8 bytes as header size
        uint64_t headerSize;
        safeTensorFile.read(reinterpret_cast<char *>(&headerSize), 8);

        // load the header
        std::string header(headerSize, '\0');
        safeTensorFile.read(header.data(), headerSize);

        // find end of the safetensor file
        uint64_t weightEndOffset{0};
        rapidjson::Document headerDoc;
        headerDoc.Parse(header.c_str());

        for (auto &[key, value] : headerDoc.GetObject())
        {
            if (key != "__metadata__")
            {
                weightEndOffset = std::max(weightEndOffset, value["data_offsets"][1].GetUint64());
            }
        }

        std::string weightsCPU(weightEndOffset, '\0');
        safeTensorFile.read(reinterpret_cast<char *>(weightsCPU.data()), weightEndOffset);

        // close the file after reading the weights
        safeTensorFile.close();

        // check GPU availability and report errors
        checkGpuAvailablity();

        // allocate and copy tensor data toGPU memory
        void *d_weights;
        cudaMalloc(&d_weights, weightEndOffset);
        cudaMemcpy(d_weights, weightsCPU.data(), weightEndOffset, cudaMemcpyHostToDevice);
        weights.data_ptr = d_weights;

        for (auto &[key, value] : headerDoc.GetObject())
        {
            if (key != "__metadata__")
            {
                Tensor tensor;
                tensor.dataType = stringToDType(value["dtype"].GetString());
                for (auto &dim : value["shape"].GetArray())
                {
                    tensor.shape.push_back(dim.GetUint64());
                }
                tensor.dataOffset = value["data_offsets"][0].GetUint64();
                tensor.dataSize = value["data_offsets"][1].GetUint64() - value["data_offsets"][0].GetUint64();
                weights.tensors[key.GetString()] = tensor;
            }
        }
    }
}
