#pragma once

#include <iostream>
#include <vector>
#include <cstdint>

namespace MiniVLLM
{
    // typedefs and structs
    enum class DataType
    {
        FLOAT32,
        FLOAT16,
        BFLOAT16,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64
    };

    struct Tensor
    {
        DataType dataType;
        std::vector<std::uint64_t> shape;
        std::uint64_t dataOffset; // offset from start pointer of all weights in GPU memory
        std::uint64_t dataSize;
    };

    // functions

    std::ostream &operator<<(std::ostream &os, DataType type);

    DataType stringToDataType(const std::string &dtype);
}
