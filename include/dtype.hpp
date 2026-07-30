#pragma once

#include <iostream>
#include <vector>
#include <map>

#include <cstdint>

namespace MiniVLLM
{
    // typedefs and structs
    enum class DType
    {
        Float32,
        Float16,
        BFloat16,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64
    };

    struct Tensor
    {
        DType dataType;
        std::vector<std::uint64_t> shape;
        std::uint64_t dataOffset; // offset from start pointer of all weights in GPU memory
        std::uint64_t dataSize;
    };

    struct Weights
    {
        void *data_ptr; // start pointer of all weights in GPU memory
        std::map<std::string, Tensor> tensors;
    };

    // functions
    std::ostream &operator<<(std::ostream &os, DType type);

    DType stringToDType(const std::string &dtype);
}
