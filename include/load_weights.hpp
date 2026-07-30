#pragma once

#include "dtype.hpp"

#include <string>
#include <map>

namespace MiniVLLM
{
    void loadWeightsFromSafeTensor(const std::string &modelFilePath,
                                   Weights &weights);
};
