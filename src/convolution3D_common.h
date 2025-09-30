#ifndef CONVOLUTION3D_COMMON_H
#define CONVOLUTION3D_COMMON_H

#include <cstddef>
#include <cassert>
#include <cuda_runtime.h>

constexpr size_t MAX_KERNEL_CONSTANT_BYTES = 64 * 1024;
constexpr int MAX_KERNEL_CONSTANT_ELEMENTS = MAX_KERNEL_CONSTANT_BYTES / sizeof(float);
constexpr size_t MAX_SHARED_MEMORY_BYTES = 64 * 1024;
constexpr size_t MAX_SHARED_MEMORY_ELEMENTS = MAX_SHARED_MEMORY_BYTES / sizeof(float);


// Reference CPU 3D convolution implementation with a single thread
void convolution3DGold(float* p_output, 
    const float* p_input, 
    const float* p_kernel,
    const int width, 
    const int height, 
    const int depth,
    const int kernel_radius_x, 
    const int kernel_radius_y, 
    const int kernel_radius_z, 
    const bool use_zero_padding);

// Upload convolution kernel to device constant memory
cudaError_t uploadConvolutionKernelToConstantMemory(const float* h_kernel,
                                                    int kernel_radius_x,
                                                    int kernel_radius_y,
                                                    int kernel_radius_z);

// Compute required shared memory size for optimized convolution kernel
size_t convolution3DSharedTileSizeBytes(dim3 block_dim,
                                        int kernel_radius_x,
                                        int kernel_radius_y,
                                        int kernel_radius_z);

// Naive convolution with global kernel (no shared/constant memory)
__global__ void convolution3DBaseline(float* __restrict__ p_output, const float* __restrict__ p_input,
    const float* __restrict__ p_kernel, const int width, const int height, const int depth, 
    const int kernel_radius_x, const int kernel_radius_y, const int kernel_radius_z, const bool use_zero_padding);

// Naive convolution without memory optimizations launcher to be called from C++ code
extern "C" void launchConvolution3DBaseline(
    const dim3 gridDim,
    const dim3 blockDim,
    float* p_output,
    const float* p_input,
    const float* p_kernel,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius_x,
    const int kernel_radius_y,
    const int kernel_radius_z,
    const bool use_zero_padding);


// Naive convolution with shared memory and constant memory for the kernel
__global__ void convolution3DOptimized(float* __restrict__ p_output, const float* __restrict__ p_input,
    const int width, const int height, const int depth, 
    const int kernel_radius_x, const int kernel_radius_y, const int kernel_radius_z);

// Optimized convolution launcher with shared memory and constant memory for the kernel to be called from C++ code
extern "C" void launchConvolution3DOptimized(
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


// Separable 3D convolution with shared memory and constant memory for the 1D kernels
cudaError_t convolution3DSeparable(
    float* d_output,
    const float* d_input,
    const float* h_kernel_x,
    const float* h_kernel_y,
    const float* h_kernel_z,
    int width,
    int height,
    int depth,
    int kernel_radius_x,
    int kernel_radius_y,
    int kernel_radius_z,
    bool use_zero_padding,
    cudaStream_t stream = nullptr,
    dim3 block_dim = dim3(0, 0, 0));

#endif