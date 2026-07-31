#include "llama.hpp"
#include "dtype.hpp"
#include "config.hpp"

namespace MiniVLLM::Llama
{

    const ModelConfig &getConfig()
    {
        static constexpr ModelConfig config = {
            .DTYPE = MiniVLLM::DType::BFloat16,

            .VOCAB_SIZE = 128256,

            .HIDDEN_SIZE = 2048,
            .NUM_ATTENTION_HEADS = 32,
            .NUM_KEY_VALUE_HEADS = 8,
            .HEAD_DIM = 64,
            .NUM_HIDDEN_LAYERS = 16,

            .INTERMEDIATE_SIZE = 8192,
            .MAX_POSITION_EMBEDDINGS = 131072,

            .RMS_NORM_EPS = 1e-5f,
            .ROPE_THETA = 500000.0f,

            .ROPE_SCALING = {
                .FACTOR = 32.0f,
                .LOW_FREQ_FACTOR = 1.0f,
                .HIGH_FREQ_FACTOR = 4.0f,
                .ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192,
            },
        };

        return config;
    }

}