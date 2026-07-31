cmake_minimum_required(VERSION 3.20)

project(TransformerInference LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------------

add_library(transformer_lib
    src/kernels_api.cu
    src/load_weights.cpp
    src/dtype.cpp
    src/utils.cpp
    src/kernels.cu
    model_config/llama.cu
)

set_target_properties(transformer_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(transformer_lib PUBLIC
    include
    model_config
)

target_link_libraries(transformer_lib PUBLIC
    CUDA::cudart
)

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------

add_executable(transformer
    src/main.cpp
)

target_link_libraries(transformer PRIVATE
    transformer_lib
)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

add_executable(test_rms_norm
    tests/test_rms_norm.cu
)

set_target_properties(test_rms_norm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(test_rms_norm PRIVATE
    transformer_lib
)

add_test(NAME rms_norm COMMAND test_rms_norm)

add_executable(test_rope
    tests/test_rope.cu
)

set_target_properties(test_rope PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(test_rope PRIVATE
    transformer_lib
)

add_test(NAME rope COMMAND test_rope)
