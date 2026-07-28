#pragma once

#include <string>
#include <vector>

namespace MiniVLLM
{
    void validateSafeTensorFileExists(const std::string &);

    void checkGpuAvailablity();

    // run python tokenizer script to tokenize the input text and return the tokens as a vector of integers
    std::vector<int> tokenize(const std::string &text);
}
