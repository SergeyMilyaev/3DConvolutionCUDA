#ifndef CONVOLUTION3D_COMMON_H
#define CONVOLUTION3D_COMMON_H

#include <cstddef>
#include <cassert>
#include <cuda_runtime.h>

constexpr size_t MAX_KERNEL_CONSTANT_BYTES = 64 * 1024;
constexpr int MAX_KERNEL_CONSTANT_ELEMENTS = MAX_KERNEL_CONSTANT_BYTES / sizeof(float);
constexpr size_t MAX_SHARED_MEMORY_BYTES = 64 * 1024;
constexpr size_t MAX_SHARED_MEMORY_ELEMENTS = MAX_SHARED_MEMORY_BYTES / sizeof(float);

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
void convolution3D_gold(float* p_output, const float* p_input, const float* p_kernel,
    const int width, const int height, const int depth, const int kernel_radius_x, const int kernel_radius_y, const int kernel_radius_z, const bool use_zero_padding);


cudaError_t uploadConvolutionKernelToConstantMemory(const float* h_kernel,
                                                    int kernel_radius_x,
                                                    int kernel_radius_y,
                                                    int kernel_radius_z);

size_t convolution3DSharedTileSizeBytes(dim3 block_dim,
                                        int kernel_radius_x,
                                        int kernel_radius_y,
                                        int kernel_radius_z);

////////////////////////////////////////////////////////////////////////////////
// Naive convolution
////////////////////////////////////////////////////////////////////////////////
__global__ void convolution3D_naive(float* __restrict__ p_output, const float* __restrict__ p_input,
    const int width, const int height, const int depth, 
    const int kernel_radius_x, const int kernel_radius_y, const int kernel_radius_z);

extern "C" void launch_convolution3D_naive(
    const dim3 gridDim,
    const dim3 blockDim,
    const size_t shared_size,
    float* p_output,
    const float* p_input,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius_x,
    const int kernel_radius_y,
    const int kernel_radius_z,
    const bool use_zero_padding);

#endif