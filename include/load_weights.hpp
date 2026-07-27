#pragma once

#include <string>
#include <map>
#include "tensor.hpp"

namespace MiniVLLM
{
    // structs

    struct Weights
    {
        void *data_ptr; // start pointer of all weights in GPU memory
        std::map<std::string, Tensor> tensors;
    };

    // functions

    void loadWeightsFromSafeTensor(const std::string &modelFilePath,
                                   Weights &weights);
};
