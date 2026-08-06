#include "config.hpp"
#include "kernels.cuh"

#include <cuda_bf16.h>

namespace MiniVLLM
{
    /**
     * @brief Applies the SiLU (Sigmoid Linear Unit) activation to the gate projection
     *        and performs element-wise multiplication with the up projection.
     *
     * Computes:
     *     gateProjection[i] = SiLU(gateProjection[i]) * upProjection[i]
     *
     * where:
     *     SiLU(x) = x / (1 + exp(-x))
     *
     * The kernel uses a grid-stride loop, allowing it to process projection
     * tensors of arbitrary size regardless of the number of launched threads.
     *
     * @tparam T Data type of the projection tensors (e.g. __half, __nv_bfloat16, float).
     *
     * @param projectionDim Total number of elements in the projection tensors.
     * @param gateProjection Input gate projection. Overwritten in-place with
     *                       SiLU(gateProjection) * upProjection.
     * @param upProjection Input up projection tensor. Remains unmodified.
     */
    template <typename T>
    __global__ void silu(int projectionDim, T *gateProjection, T *upProjection)
    {
        int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
        int stride = gridDim.x * THREADS_PER_BLOCK;

        for (int i = idx; i < projectionDim; i += stride)
        {
            float gate = static_cast<float>(gateProjection[i]);
            float up = static_cast<float>(upProjection[i]);

            float silu = gate / (1.0f + expf(-gate));

            gateProjection[i] = static_cast<T>(silu * up);
        }
    }

    template __global__ void silu<__nv_bfloat16>(int projectionDim, __nv_bfloat16 *gateProjection, __nv_bfloat16 *upProjection);
    template __global__ void silu<float>(int projectionDim, float *gateProjection, float *upProjection);
}