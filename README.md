# mini-vllm

A high-performance LLM inference engine built from scratch with C++ and CUDA. mini-vllm is designed as a younger and smaller sibling of vLLM, built to maximize efficient use of hardware for high throughput and low latency.

## Features

The inference engine aims to support the following:
- [x] Load a real LLM model from Safetensors (modular, runs on every model)
- [x] All computation with CUDA kernels
- [ ] Full LLM forward pass (prefill + decode)
- [ ] KV cache
- [ ] Static batching
- [ ] Continuous batching
- [ ] Online softmax, FlashAttention-like
- [ ] PagedAttention

## Technical Prerequisites

You can build and run this project on any platform with an NVIDIA GPU (CUDA is required). 

### Development Environment (Tested Setup)
- Linux (Ubuntu 25.10, Kernel 6.17.0-41-generic)
- CUDA Toolkit (13.3)
- C++ 17
- GCC (15.2.0)
- External Dependency: JSON parser `rapidjson` (`include/rapidjson`)
- Intel CPU (Core Ultra 9 275HX @ 5.40 GHz)
- NVIDIA GPU (RTX 5080 Max-Q / Mobile)

## Installation and Setup

### 1. Prerequisites
- Ensure you have **CMake** installed on your system.
- Python dependencies (for the tokenizer): `pip install transformers`

### 2. Getting the Model
You will need to clone the complete model repository. For development, we use [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). 

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct models/Llama-3.2-1B
```
The entire model directory is required (not just the safetensors file) because the Python tokenizer depends on the full model repository. Ensure the model path is correctly set or cloned into `models/<model_name>` (e.g., `models/Llama-3.2-1B`).

### 3. Build and Run

The project includes a single command-line utility for building, running, and cleaning the project.

#### Build
Configure and compile the project from scratch:

```bash
./minivllm build
```

#### Run
Execute the compiled binary:

```bash
./minivllm run
```

#### Clean
Remove the build directory and all generated files:

```bash
./minivllm clean
```

> **Note:** The `build` command always removes the existing `build/` directory before configuring and compiling a fresh build.

You may need to adjust environment-specific paths, such as CUDA or GCC in `c_cpp_properties.json`, or the CUDA compiler configuration in `CMakeLists.txt`.

## Implementation Progress

Here is the checklist of components and kernels needed for a complete LLM inference pipeline:

- [x] Safetensors model loading
- [x] Single token inference structure
- [x] Tokenization
- [x] Embeddings computation
- [x] CUDA kernel engineering - embeddings
- [x] RMSNorm and parallel reduction in CUDA
- [ ] RoPE (Rotary Positional Embeddings)
- [ ] Residual connections
- [ ] cuBLAS GemmEx matrix multiplications
- [ ] Attention mechanism
- [ ] GQA (Grouped-Query Attention)
- [ ] SiLU activation function
- [ ] Softmax (Online softmax)
- [ ] Causal masking
- [ ] Argmax sampling
- [ ] Feed Forward Network
- [ ] KV cache management
- [ ] Static batching
- [ ] Continuous batching
- [ ] Paged Attention
- [ ] Paged KV cache
- [ ] Paged Attention CUDA kernel
