#include "convolution3D_common.h"

static_assert(MAX_SHARED_MEMORY_BYTES == 64 * 1024, "Shared memory limit mismatch");
static_assert(MAX_KERNEL_CONSTANT_BYTES == 64 * 1024, "Constant memory limit mismatch");

__constant__ float c_kernel[MAX_KERNEL_CONSTANT_ELEMENTS];

namespace {

__device__ __forceinline__ int flatten3D(const int x, const int y, const int z,
                                         const int dim_x, const int dim_y) {
    return z * dim_y * dim_x + y * dim_x + x;
}


__device__ __forceinline__ int clamp(int x, int a, int b) {
    return max(a, min(x, b - 1));
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


__global__ void convolution3DBaseline(
    float* __restrict__ p_output,
    const float* __restrict__ p_input,
    const float* __restrict__ p_kernel,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius_x,
    const int kernel_radius_y,
    const int kernel_radius_z,
    const bool use_zero_padding)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) {
        return;
    }

    const int kernel_dim_x = 2 * kernel_radius_x + 1;
    const int kernel_dim_y = 2 * kernel_radius_y + 1;
    const int kernel_dim_z = 2 * kernel_radius_z + 1;

    float sum = 0.0f;

    // Iterate over the kernel
    for (int kz = 0; kz < kernel_dim_z; ++kz) {
        const int input_z = z + kz - kernel_radius_z;        
        for (int ky = 0; ky < kernel_dim_y; ++ky) {
            const int input_y = y + ky - kernel_radius_y;            
            for (int kx = 0; kx < kernel_dim_x; ++kx) {
                const int input_x = x + kx - kernel_radius_x;
                
                // Handle boundary conditions
                const int clamped_x = clamp(input_x, 0, width);
                const int clamped_y = clamp(input_y, 0, height);
                const int clamped_z = clamp(input_z, 0, depth);
                const bool is_in_bounds = (input_x == clamped_x) && (input_y == clamped_y) && (input_z == clamped_z);

                // Load input and kernel values
                const int input_idx = flatten3D(clamped_x, clamped_y, clamped_z, width, height);
                const float input_val = p_input[input_idx] * static_cast<float>(is_in_bounds || !use_zero_padding);
                const int kernel_idx = flatten3D(kx, ky, kz, kernel_dim_x, kernel_dim_y);
                const float kernel_val = p_kernel[kernel_idx];

                sum += input_val * kernel_val;
            }
        }
    }

    const int output_idx = flatten3D(x, y, z, width, height);
    p_output[output_idx] = sum;
}


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
    const bool use_zero_padding)
{
    convolution3DBaseline<<<gridDim, blockDim>>>(
        p_output, p_input, p_kernel, width, height, depth,
        kernel_radius_x, kernel_radius_y, kernel_radius_z, use_zero_padding);
}


__global__ void convolution3DOptimized(
    float* __restrict__ p_output,
    const float* __restrict__ p_input,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius_x,
    const int kernel_radius_y,
    const int kernel_radius_z,
    const bool use_zero_padding)
{
    
    // Dynamically allocated shared memory. Size is passed at kernel launch.
    extern __shared__ float s_tile_1d[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int block_base_x = blockIdx.x * blockDim.x;
    const int block_base_y = blockIdx.y * blockDim.y;
    const int block_base_z = blockIdx.z * blockDim.z;

    // Global position of the start of the output tile this block computes
    const int x = block_base_x + tx;
    const int y = block_base_y + ty;
    const int z = block_base_z + tz;

    if (x >= width || y >= height || z >= depth) {
        return;
    }

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

    // Load data into shared memory
    for (int sz = tz; sz < shared_dim_z; sz += tile_dim_z) {
        const int global_z = block_base_z + sz - kernel_radius_z;
        for (int sy = ty; sy < shared_dim_y; sy += tile_dim_y) {
            const int global_y = block_base_y + sy - kernel_radius_y;
            for (int sx = tx; sx < shared_dim_x; sx += tile_dim_x) {
                const int global_x = block_base_x + sx - kernel_radius_x;
                // Handle boundary conditions
                const int shared_idx = sz * shared_stride_z + sy * shared_stride_y + sx;
                const int src_x = clamp(global_x, 0, width);
                const int src_y = clamp(global_y, 0, height);
                const int src_z = clamp(global_z, 0, depth);
                const int global_idx = flatten3D(src_x, src_y, src_z, width, height);
                const bool is_in_bounds = (global_x == src_x) && (global_y == src_y) && (global_z == src_z);

                s_tile_1d[shared_idx] = p_input[global_idx] * static_cast<float>(is_in_bounds || !use_zero_padding);
            }
        }
    }

    __syncthreads();
    
    // Compute convolution
    if (tx < tile_dim_x && ty < tile_dim_y && tz < tile_dim_z) {
        float sum = 0.0f;
        // Iterate over the kernel and saved tile
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

        // Write the result
        if (output_x < width && output_y < height && output_z < depth) {
            const int output_idx = flatten3D(output_x, output_y, output_z, width, height);
            p_output[output_idx] = sum;
        }
    }
}


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
    const bool use_zero_padding)
{
    convolution3DOptimized<<<gridDim, blockDim, shared_size>>>(
        p_output, p_input, width, height, depth,
        kernel_radius_x, kernel_radius_y, kernel_radius_z, use_zero_padding);
}


__global__ void convolutionRowX(
    float* p_output,
    const float* p_input,
    const float* p_kernel,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius_x,
    const bool use_zero_padding)
{
    // Dynamically allocated shared memory. Size is passed at kernel launch.
    extern __shared__ float s_tile_1d[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int block_base_x = blockIdx.x * blockDim.x;
    const int block_base_y = blockIdx.y * blockDim.y;
    const int block_base_z = blockIdx.z * blockDim.z;

    // Global position of the start of the output tile this block computes
    const int x = block_base_x + tx;
    const int y = block_base_y + ty;
    const int z = block_base_z + tz;

    if (x >= width || y >= height || z >= depth) {
        return;
    }
    
    const int tile_dim_x = min(blockDim.x, width - block_base_x);
    const int tile_dim_y = min(blockDim.y, height - block_base_y);
    const int tile_dim_z = min(blockDim.z, depth - block_base_z);

    const int kernel_dim_x = 2 * kernel_radius_x + 1;
    
    // Width of the shared memory tile, including apron
    const int shared_width = blockDim.x + 2 * kernel_radius_x;
    const int shared_stride_z = shared_width * tile_dim_y;

    // Load data into shared memory
    for (int sx = tx; sx < shared_width; sx += tile_dim_x) {
        const int shared_idx = tz * shared_stride_z + ty * shared_width + sx;
        const int global_x = block_base_x + sx - kernel_radius_x;
        const int src_x = clamp(global_x, 0, width);
        const bool is_in_bounds = (global_x == src_x);
        const int global_idx = flatten3D(src_x, y, z, width, height);
        s_tile_1d[shared_idx] = p_input[global_idx] * static_cast<float>(is_in_bounds || !use_zero_padding);
    }

    __syncthreads();

    // Compute convolution
    if (tx < tile_dim_x && ty < tile_dim_y && tz < tile_dim_z) {
        float sum = 0.0f;
        // Iterate over the kernel and saved tile
        for (int kx = 0; kx < kernel_dim_x; ++kx) {
            const int tile_x = tx + kx;
            const int tile_idx = tz * shared_stride_z + ty * shared_width + tile_x;
            sum += s_tile_1d[tile_idx] * c_kernel[kx];
        }

        const int output_x = block_base_x + tx;
        const int output_y = block_base_y + ty;
        const int output_z = block_base_z + tz;

        // Write the result
        if (output_x < width && output_y < height && output_z < depth) {
            const int output_idx = flatten3D(output_x, output_y, output_z, width, height);
            p_output[output_idx] = sum;
        }
    }
}


__global__ void convolutionRowY(
    float* p_output,
    const float* p_input,
    const float* p_kernel,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius_y,
    const bool use_zero_padding)
{
    // Dynamically allocated shared memory. Size is passed at kernel launch.
    extern __shared__ float s_tile_1d[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int block_base_x = blockIdx.x * blockDim.x;
    const int block_base_y = blockIdx.y * blockDim.y;
    const int block_base_z = blockIdx.z * blockDim.z;

    // Global position of the start of the output tile this block computes
    const int x = block_base_x + tx;
    const int y = block_base_y + ty;
    const int z = block_base_z + tz;

    if (x >= width || y >= height || z >= depth) {
        return;
    }

    const int tile_dim_x = min(blockDim.x, width - block_base_x);
    const int tile_dim_y = min(blockDim.y, height - block_base_y);
    const int tile_dim_z = min(blockDim.z, depth - block_base_z);

    const int kernel_dim_y = 2 * kernel_radius_y + 1;
    
    // Height of the shared memory tile, including apron
    const int shared_height = blockDim.y + 2 * kernel_radius_y;
    const int shared_stride_z = tile_dim_x * shared_height;

    // Load data into shared memory
    for (int sy = ty; sy < shared_height; sy += tile_dim_y) {
        const int shared_idx = tz * shared_stride_z + sy * tile_dim_x + tx;
        const int global_y = block_base_y + sy - kernel_radius_y;
        const int src_y = clamp(global_y, 0, height);
        const bool is_in_bounds = (global_y == src_y);
        const int global_idx = flatten3D(x, src_y, z, width, height);
        s_tile_1d[shared_idx] = p_input[global_idx] * static_cast<float>(is_in_bounds || !use_zero_padding);
    }

    __syncthreads();

    // Compute convolution
    if (tx < tile_dim_x && ty < tile_dim_y && tz < tile_dim_z) {
        float sum = 0.0f;
        // Iterate over the kernel and saved tile
        for (int ky = 0; ky < kernel_dim_y; ++ky) {
            const int tile_y = ty + ky;
            const int tile_idx = tz * shared_stride_z + tile_y * tile_dim_x + tx;
            sum += s_tile_1d[tile_idx] * c_kernel[ky];
        }

        const int output_x = block_base_x + tx;
        const int output_y = block_base_y + ty;
        const int output_z = block_base_z + tz;

        // Write the result
        if (output_x < width && output_y < height && output_z < depth) {
            const int output_idx = flatten3D(output_x, output_y, output_z, width, height);
            p_output[output_idx] = sum;
        }
    }
}


__global__ void convolutionRowZ(
    float* p_output,
    const float* p_input,
    const float* p_kernel,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius_z,
    const bool use_zero_padding)
{
    // Dynamically allocated shared memory. Size is passed at kernel launch.
    extern __shared__ float s_tile_1d[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int block_base_x = blockIdx.x * blockDim.x;
    const int block_base_y = blockIdx.y * blockDim.y;
    const int block_base_z = blockIdx.z * blockDim.z;

    // Global position of the start of the output tile this block computes
    const int x = block_base_x + tx;
    const int y = block_base_y + ty;
    const int z = block_base_z + tz;
    if (x >= width || y >= height || z >= depth) {
        return;
    }

    const int tile_dim_x = min(blockDim.x, width - block_base_x);
    const int tile_dim_y = min(blockDim.y, height - block_base_y);
    const int tile_dim_z = min(blockDim.z, depth - block_base_z);

    const int kernel_dim_z = 2 * kernel_radius_z + 1;
    
    // Depth of the shared memory tile, including apron
    const int shared_depth = blockDim.z + 2 * kernel_radius_z;
    const int shared_stride_z = tile_dim_x * tile_dim_y;

    // Load data into shared memory
    for (int sz = tz; sz < shared_depth; sz += tile_dim_z) {
        const int shared_idx = sz * shared_stride_z + ty * tile_dim_x + tx;
        const int global_z = block_base_z + sz - kernel_radius_z;
        const int src_z = clamp(global_z, 0, depth);
        const bool is_in_bounds = (global_z == src_z);
        const int global_idx = flatten3D(x, y, src_z, width, height);
        s_tile_1d[shared_idx] = p_input[global_idx] * static_cast<float>(is_in_bounds || !use_zero_padding);
    }

    __syncthreads();

    // Compute convolution
    if (tx < tile_dim_x && ty < tile_dim_y && tz < tile_dim_z) {
        float sum = 0.0f;
        // Iterate over the kernel and saved tile
        for (int kz = 0; kz < kernel_dim_z; ++kz) {
            const int tile_z = tz + kz;
            const int tile_idx = tile_z * shared_stride_z + ty * tile_dim_x + tx;
            sum += s_tile_1d[tile_idx] * c_kernel[kz];
        }

        const int output_x = block_base_x + tx;
        const int output_y = block_base_y + ty;
        const int output_z = block_base_z + tz;

        // Write the result
        if (output_x < width && output_y < height && output_z < depth) {
            const int output_idx = flatten3D(output_x, output_y, output_z, width, height);
            p_output[output_idx] = sum;
        }
    }
}

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
    cudaStream_t stream,
    dim3 requested_block_dim)
{
    if (d_output == nullptr || d_input == nullptr ||
        h_kernel_x == nullptr || h_kernel_y == nullptr || h_kernel_z == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (width <= 0 || height <= 0 || depth <= 0 ||
        kernel_radius_x < 0 || kernel_radius_y < 0 || kernel_radius_z < 0) {
        return cudaErrorInvalidValue;
    }

    const size_t volume_elements = static_cast<size_t>(width) * height * depth;
    if (volume_elements == 0) {
        return cudaErrorInvalidValue;
    }

    const size_t volume_bytes = volume_elements * sizeof(float);

    // Allocate temporary buffers to hold intermediate results from separable filtering
    float* d_temp_1 = nullptr;
    float* d_temp_2 = nullptr;

    cudaError_t err = cudaMalloc(&d_temp_1, volume_bytes);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMalloc(&d_temp_2, volume_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_temp_1);
        return err;
    }

    auto cleanup = [&](cudaError_t status) {
        cudaFree(d_temp_1);
        cudaFree(d_temp_2);
        return status;
    };

    auto clamp_block_dim = [](int extent) -> unsigned int {
        if (extent <= 0) {
            return 1u;
        }
        const int capped = min(extent, 8);
        return static_cast<unsigned int>(capped);
    };

    // Determine block dimensions
    dim3 block_dim;
    block_dim.x = requested_block_dim.x > 0 ? requested_block_dim.x : clamp_block_dim(width);
    block_dim.y = requested_block_dim.y > 0 ? requested_block_dim.y : clamp_block_dim(height);
    block_dim.z = requested_block_dim.z > 0 ? requested_block_dim.z : clamp_block_dim(depth);

    if (block_dim.x == 0 || block_dim.y == 0 || block_dim.z == 0) {
        return cleanup(cudaErrorInvalidConfiguration);
    }

    const unsigned long long threads_per_block =
        static_cast<unsigned long long>(block_dim.x) *
        static_cast<unsigned long long>(block_dim.y) *
        static_cast<unsigned long long>(block_dim.z);
    if (threads_per_block == 0 || threads_per_block > 1024ULL) {
        return cleanup(cudaErrorInvalidConfiguration);
    }

    // Determine grid dimensions
    const unsigned int grid_x = (static_cast<unsigned int>(width) + block_dim.x - 1) / block_dim.x;
    const unsigned int grid_y = (static_cast<unsigned int>(height) + block_dim.y - 1) / block_dim.y;
    const unsigned int grid_z = (static_cast<unsigned int>(depth) + block_dim.z - 1) / block_dim.z;

    const dim3 grid_dim(grid_x, grid_y, grid_z);

    auto ensure_shared_within_limit = [](size_t bytes) -> bool {
        return bytes <= MAX_SHARED_MEMORY_BYTES;
    };

    const size_t shared_x_elements = (static_cast<size_t>(block_dim.x) + static_cast<size_t>(2 * kernel_radius_x)) *
                                     static_cast<size_t>(block_dim.y) * static_cast<size_t>(block_dim.z);
    const size_t shared_x_bytes = shared_x_elements * sizeof(float);
    if (!ensure_shared_within_limit(shared_x_bytes)) {
        return cleanup(cudaErrorInvalidConfiguration);
    }

    // Upload the 1D kernel for X-dimension to constant memory
    err = uploadConvolutionKernelToConstantMemory(h_kernel_x, kernel_radius_x, 0, 0);
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    // Launch convolution along X dimension
    convolutionRowX<<<grid_dim, block_dim, shared_x_bytes, stream>>>(
        d_temp_1,
        d_input,
        nullptr,
        width,
        height,
        depth,
        kernel_radius_x,
        use_zero_padding);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    const size_t shared_y_elements = (static_cast<size_t>(block_dim.y) + static_cast<size_t>(2 * kernel_radius_y)) *
                                     static_cast<size_t>(block_dim.x) * static_cast<size_t>(block_dim.z);
    const size_t shared_y_bytes = shared_y_elements * sizeof(float);
    if (!ensure_shared_within_limit(shared_y_bytes)) {
        return cleanup(cudaErrorInvalidConfiguration);
    }

    // Upload the 1D kernel for Y-dimension to constant memory
    err = uploadConvolutionKernelToConstantMemory(h_kernel_y, 0, kernel_radius_y, 0);
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    // Launch convolution along Y dimension
    convolutionRowY<<<grid_dim, block_dim, shared_y_bytes, stream>>>(
        d_temp_2,
        d_temp_1,
        nullptr,
        width,
        height,
        depth,
        kernel_radius_y,
        use_zero_padding);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    const size_t shared_z_elements = (static_cast<size_t>(block_dim.z) + static_cast<size_t>(2 * kernel_radius_z)) *
                                     static_cast<size_t>(block_dim.x) * static_cast<size_t>(block_dim.y);
    const size_t shared_z_bytes = shared_z_elements * sizeof(float);
    if (!ensure_shared_within_limit(shared_z_bytes)) {
        return cleanup(cudaErrorInvalidConfiguration);
    }

    // Upload the 1D kernel for Z-dimension to constant memory
    err = uploadConvolutionKernelToConstantMemory(h_kernel_z, 0, 0, kernel_radius_z);
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    // Launch convolution along Z dimension
    convolutionRowZ<<<grid_dim, block_dim, shared_z_bytes, stream>>>(
        d_output,
        d_temp_2,
        nullptr,
        width,
        height,
        depth,
        kernel_radius_z,
        use_zero_padding);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return cleanup(err);
    }

    return cleanup(cudaSuccess);
}