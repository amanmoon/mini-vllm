cmake_minimum_required(VERSION 3.20)

project(TransformerInference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(CUDAToolkit REQUIRED)

add_executable(transformer
    src/main.cu
    src/load_weights.cpp
    src/tensor.cpp
    src/utils.cpp
    src/kernels.cu
)

set_target_properties(transformer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(transformer PRIVATE
    include
)

target_link_libraries(transformer PRIVATE CUDA::cudart)
