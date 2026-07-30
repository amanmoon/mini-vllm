cmake_minimum_required(VERSION 3.20)

project(TransformerInference LANGUAGES CXX CUDA)

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
    src/kernels_api.cu
)

set_target_properties(transformer_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(transformer_lib PUBLIC
    include
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

target_link_libraries(test_rms_norm PRIVATE
    transformer_lib
)