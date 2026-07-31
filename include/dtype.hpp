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

    struct RopeScaling
    {
        float FACTOR;
        float LOW_FREQ_FACTOR;
        float HIGH_FREQ_FACTOR;
        size_t ORIGINAL_MAX_POSITION_EMBEDDINGS;
    };

    struct ModelConfig
    {
        DType DTYPE;

        size_t VOCAB_SIZE;

        size_t HIDDEN_SIZE;
        size_t NUM_ATTENTION_HEADS;
        size_t NUM_KEY_VALUE_HEADS;
        size_t HEAD_DIM;
        size_t NUM_HIDDEN_LAYERS;

        size_t INTERMEDIATE_SIZE;
        size_t MAX_POSITION_EMBEDDINGS;

        float RMS_NORM_EPS;
        float ROPE_THETA;

        RopeScaling ROPE_SCALING;
    };

    // functions
    std::ostream &operator<<(std::ostream &os, DType type);

    DType stringToDType(const std::string &dtype);
}
