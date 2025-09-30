#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>

#include "convolution3D_common.h"


// Helper to check for CUDA errors
#define CUDA_CHECK(err) { \
    cudaError_t e = err; \
    if (e!= cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void print_stats(const std::string& name, const std::vector<float>& timings) {
    if (timings.empty()) return;
    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / timings.size();
    
    std::vector<float> sorted_timings = timings;
    std::sort(sorted_timings.begin(), sorted_timings.end());
    
    double median = sorted_timings[sorted_timings.size() / 2];
    double min_val = sorted_timings[0];
    
    std::cout << name << ":" << std::endl;
    std::cout << "  Average: " << mean << " ms" << std::endl;
    std::cout << "  Median:  " << median << " ms" << std::endl;
    std::cout << "  Min:     " << min_val << " ms" << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

int main() {
    // --- Benchmark Configuration ---
    const int width = 128;
    const int height = 128;
    const int depth = 8;
    const int kernel_radius = 7;
    const int num_iterations = 20;
    const int warmup_iterations = 5;

    const size_t vol_size = (size_t)width * height * depth;
    const size_t vol_bytes = vol_size * sizeof(float);
    
    const int kernel_size = 2 * kernel_radius + 1;
    const size_t kernel_vol = (size_t)kernel_size * kernel_size * kernel_size;
    const size_t kernel_bytes = kernel_vol * sizeof(float);

    // --- Host Data Allocation and Initialization ---
    std::vector<float> h_input(vol_size);
    std::vector<float> h_kernel(kernel_vol);
    std::vector<float> h_kernel_x(kernel_size);
    std::vector<float> h_kernel_y(kernel_size);
    std::vector<float> h_kernel_z(kernel_size);
    std::vector<float> h_output_cpu(vol_size);
    std::vector<float> h_output_naive(vol_size);
    std::vector<float> h_output_naive_global(vol_size);
    std::vector<float> h_output_separable(vol_size);

    for(size_t i = 0; i < vol_size; ++i) 
        h_input[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < kernel_size; ++i) {
        h_kernel_x[i] = static_cast<float>(rand()) / RAND_MAX;
        h_kernel_y[i] = static_cast<float>(rand()) / RAND_MAX;
        h_kernel_z[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int kz = 0; kz < kernel_size; ++kz) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int idx = kz * kernel_size * kernel_size + ky * kernel_size + kx;
                h_kernel[idx] = h_kernel_x[kx] * h_kernel_y[ky] * h_kernel_z[kz];
            }
        }
    }

    auto measure_cpu_iteration = [&]() -> float {
        const auto start = std::chrono::steady_clock::now();
        convolution3D_gold(h_output_cpu.data(), h_input.data(), h_kernel.data(),
                           width, height, depth,
                           kernel_radius, kernel_radius, kernel_radius,
                           false);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<float, std::milli> elapsed = end - start;
        return elapsed.count();
    };

    std::cout << "=== CPU Reference (convolution3D_gold) ===" << std::endl;
    for (int i = 0; i < warmup_iterations; ++i) {
        const float ms = measure_cpu_iteration();
        std::cout << "Warmup iteration " << (i + 1) << "/" << warmup_iterations
                  << " -> " << ms << " ms" << std::endl;
    }

    std::vector<float> cpu_timings;
    cpu_timings.reserve(num_iterations);
    for (int i = 0; i < num_iterations; ++i) {
        const float ms = measure_cpu_iteration();
        cpu_timings.push_back(ms);
        std::cout << "Iteration " << (i + 1) << "/" << num_iterations
                  << " -> " << ms << " ms" << std::endl;
    }
    print_stats("CPU convolution3D_gold", cpu_timings);

    auto max_abs_difference = [&](const std::vector<float>& lhs, const std::vector<float>& rhs) {
        double max_diff = 0.0;
        for (size_t idx = 0; idx < lhs.size(); ++idx) {
            const double diff = std::abs(static_cast<double>(lhs[idx]) -
                                         static_cast<double>(rhs[idx]));
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        return max_diff;
    };

    // --- GPU Data Allocation ---
    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), vol_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(uploadConvolutionKernelToConstantMemory(
        h_kernel.data(), kernel_radius, kernel_radius, kernel_radius));

    const dim3 block_dim(8, 8, 8);
    const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                        (height + block_dim.y - 1) / block_dim.y,
                        (depth + block_dim.z - 1) / block_dim.z);
    const size_t shared_size = convolution3DSharedTileSizeBytes(
        block_dim, kernel_radius, kernel_radius, kernel_radius);

    cudaEvent_t gpu_start, gpu_stop;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));

    auto time_cuda_launch = [&](const std::function<void(void)>& launch) -> float {
        CUDA_CHECK(cudaEventRecord(gpu_start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(gpu_stop));
        CUDA_CHECK(cudaEventSynchronize(gpu_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, gpu_start, gpu_stop));
        return ms;
    };

    auto benchmark_cuda_kernel = [&](const std::string& label,
                                     const std::function<void(void)>& launch,
                                     std::vector<float>& timings,
                                     std::vector<float>& host_output) {
        std::cout << "=== " << label << " ===" << std::endl;
        for (int i = 0; i < warmup_iterations; ++i) {
            const float ms = time_cuda_launch(launch);
            std::cout << "Warmup iteration " << (i + 1) << "/" << warmup_iterations
                      << " -> " << ms << " ms" << std::endl;
        }

        timings.clear();
        timings.reserve(num_iterations);
        for (int i = 0; i < num_iterations; ++i) {
            const float ms = time_cuda_launch(launch);
            timings.push_back(ms);
            std::cout << "Iteration " << (i + 1) << "/" << num_iterations
                      << " -> " << ms << " ms" << std::endl;
        }
        print_stats(label, timings);

        CUDA_CHECK(cudaMemcpy(host_output.data(), d_output, vol_bytes, cudaMemcpyDeviceToHost));
        std::cout << "Max absolute difference |CPU - " << label << "|: "
                  << max_abs_difference(h_output_cpu, host_output)
                  << std::endl;
    };

    const std::function<void(void)> launch_naive = [&]() {
        launch_convolution3D_naive(grid_dim, block_dim, shared_size,
                                   d_output, d_input,
                                   width, height, depth,
                                   kernel_radius, kernel_radius, kernel_radius,
                                   false);
    };

    const std::function<void(void)> launch_naive_global = [&]() {
        launch_convolution3D_naive_global(grid_dim, block_dim,
                                          d_output, d_input, d_kernel,
                                          width, height, depth,
                                          kernel_radius, kernel_radius, kernel_radius,
                                          false);
    };

    const std::function<void(void)> launch_separable = [&]() {
        CUDA_CHECK(convolution3DSeparable(d_output, d_input,
                                          h_kernel_x.data(), h_kernel_y.data(), h_kernel_z.data(),
                                          width, height, depth,
                                          kernel_radius, kernel_radius, kernel_radius,
                                          false,
                                          nullptr,
                                          block_dim));
    };

    std::vector<float> timings_naive;
    benchmark_cuda_kernel("CUDA convolution3D_naive",
                          launch_naive,
                          timings_naive,
                          h_output_naive);

    std::vector<float> timings_naive_global;
    benchmark_cuda_kernel("CUDA convolution3D_naive_global",
                          launch_naive_global,
                          timings_naive_global,
                          h_output_naive_global);

    std::vector<float> timings_separable;
    benchmark_cuda_kernel("CUDA convolution3DSeparable",
                          launch_separable,
                          timings_separable,
                          h_output_separable);

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
    
    return 0;
}