#include "convolution3D_common.h"

static_assert(MAX_SHARED_MEMORY_BYTES == 64 * 1024, "Shared memory limit mismatch");
static_assert(MAX_KERNEL_CONSTANT_BYTES == 64 * 1024, "Constant memory limit mismatch");

__constant__ float c_kernel[MAX_KERNEL_CONSTANT_ELEMENTS];

namespace {

__device__ __forceinline__ int flatten3D(const int x, const int y, const int z,
                                         const int dim_x, const int dim_y) {
    return z * dim_y * dim_x + y * dim_x + x;
}

}  // namespace

cudaError_t uploadConvolutionKernelToConstantMemory(const float* h_kernel,
                                                   const int kernel_radius_x,
                                                   const int kernel_radius_y,
                                                   const int kernel_radius_z) {
    if (h_kernel == nullptr || kernel_radius_x < 0 || kernel_radius_y < 0 || kernel_radius_z < 0) {
        return cudaErrorInvalidValue;
    }

    const int kernel_dim_x = 2 * kernel_radius_x + 1;
    const int kernel_dim_y = 2 * kernel_radius_y + 1;
    const int kernel_dim_z = 2 * kernel_radius_z + 1;

    const size_t kernel_volume = static_cast<size_t>(kernel_dim_x) * kernel_dim_y * kernel_dim_z;
    const size_t kernel_bytes = kernel_volume * sizeof(float);

    if (kernel_bytes > MAX_KERNEL_CONSTANT_BYTES) {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpyToSymbol(c_kernel, h_kernel, kernel_bytes, 0, cudaMemcpyHostToDevice);
}

size_t convolution3DSharedTileSizeBytes(const dim3 block_dim,
                                        const int kernel_radius_x,
                                        const int kernel_radius_y,
                                        const int kernel_radius_z) {
    if (kernel_radius_x < 0 || kernel_radius_y < 0 || kernel_radius_z < 0) {
        return 0;
    }

    const int kernel_dim_x = 2 * kernel_radius_x + 1;
    const int kernel_dim_y = 2 * kernel_radius_y + 1;
    const int kernel_dim_z = 2 * kernel_radius_z + 1;

    const int shared_dim_x = static_cast<int>(block_dim.x) + kernel_dim_x - 1;
    const int shared_dim_y = static_cast<int>(block_dim.y) + kernel_dim_y - 1;
    const int shared_dim_z = static_cast<int>(block_dim.z) + kernel_dim_z - 1;

    const size_t shared_elements = static_cast<size_t>(shared_dim_x) * shared_dim_y * shared_dim_z;
    const size_t shared_bytes = shared_elements * sizeof(float);

    assert(shared_bytes <= MAX_SHARED_MEMORY_BYTES && "Shared memory tile exceeds 64 KB limit");
    return shared_bytes;
}

__global__ void convolution3D_naive(
    float* __restrict__ p_output,
    const float* __restrict__ p_input,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius_x,
    const int kernel_radius_y,
    const int kernel_radius_z)
{
    extern __shared__ float s_tile_1d[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int z = blockIdx.z * blockDim.z + tz;
    if (x >= width || y >= height || z >= depth) {
        return;
    }

    const int block_base_x = blockIdx.x * blockDim.x;
    const int block_base_y = blockIdx.y * blockDim.y;
    const int block_base_z = blockIdx.z * blockDim.z;

    const int tile_dim_x = min(blockDim.x, width - block_base_x);
    const int tile_dim_y = min(blockDim.y, height - block_base_y);
    const int tile_dim_z = min(blockDim.z, depth - block_base_z);

    const int kernel_dim_x = 2 * kernel_radius_x + 1;
    const int kernel_dim_y = 2 * kernel_radius_y + 1;
    const int kernel_dim_z = 2 * kernel_radius_z + 1;

    const int shared_dim_x = tile_dim_x + kernel_dim_x - 1;
    const int shared_dim_y = tile_dim_y + kernel_dim_y - 1;
    const int shared_dim_z = tile_dim_z + kernel_dim_z - 1;

    const int shared_stride_y = shared_dim_x;
    const int shared_stride_z = shared_dim_x * shared_dim_y;

    for (int sz = tz; sz < shared_dim_z; sz += tile_dim_z) {
        const int global_z = block_base_z + sz - kernel_radius_z;
        for (int sy = ty; sy < shared_dim_y; sy += tile_dim_y) {
            const int global_y = block_base_y + sy - kernel_radius_y;
            for (int sx = tx; sx < shared_dim_x; sx += tile_dim_x) {
                const int global_x = block_base_x + sx - kernel_radius_x;
                const int shared_idx = sz * shared_stride_z + sy * shared_stride_y + sx;

                if (global_x >= 0 && global_x < width &&
                    global_y >= 0 && global_y < height &&
                    global_z >= 0 && global_z < depth)
                {
                    const int global_idx = flatten3D(global_x, global_y, global_z, width, height);
                    s_tile_1d[shared_idx] = p_input[global_idx];
                } else {
                    s_tile_1d[shared_idx] = 0.0f;
                }
            }
        }
    }

    __syncthreads();

    if (tx < tile_dim_x && ty < tile_dim_y && tz < tile_dim_z) {
        float sum = 0.0f;
        for (int kz = 0; kz < kernel_dim_z; ++kz) {
            for (int ky = 0; ky < kernel_dim_y; ++ky) {
                for (int kx = 0; kx < kernel_dim_x; ++kx) {
                    const int kernel_idx = flatten3D(kx, ky, kz, kernel_dim_x, kernel_dim_y);
                    const int tile_x = tx + kx;
                    const int tile_y = ty + ky;
                    const int tile_z = tz + kz;
                    const int tile_idx = tile_z * shared_stride_z + tile_y * shared_stride_y + tile_x;
                    sum += s_tile_1d[tile_idx] * c_kernel[kernel_idx];
                }
            }
        }

        const int output_x = block_base_x + tx;
        const int output_y = block_base_y + ty;
        const int output_z = block_base_z + tz;

        if (output_x < width && output_y < height && output_z < depth) {
            const int output_idx = flatten3D(output_x, output_y, output_z, width, height);
            p_output[output_idx] = sum;
        }
    }
}


// Wrapper function to be called from C++ code
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
    const int kernel_radius_z)
{
    convolution3D_naive<<<gridDim, blockDim, shared_size>>>(
        p_output, p_input, width, height, depth,
        kernel_radius_x, kernel_radius_y, kernel_radius_z);
}