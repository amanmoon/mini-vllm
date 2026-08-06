#pragma once

namespace MiniVLLM
{
    inline constexpr int THREADS_PER_BLOCK = 1024;
    constexpr int WARP_SIZE = 32;

    constexpr float HUGE_VALUE_FLOAT = 1e9f;
}
