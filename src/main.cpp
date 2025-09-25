#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
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
    const int kernel_radius = 8;
    const int num_iterations = 100;
    const int warmup_iterations = 10;

    const size_t vol_size = (size_t)width * height * depth;
    const size_t vol_bytes = vol_size * sizeof(float);
    
    const int kernel_size = 2 * kernel_radius + 1;
    const size_t kernel_vol = (size_t)kernel_size * kernel_size * kernel_size;
    const size_t kernel_bytes = kernel_vol * sizeof(float);
    const size_t sep_kernel_bytes = kernel_size * sizeof(float);

    // --- Host Data Allocation and Initialization ---
    std::vector<float> h_input(vol_size);
    std::vector<float> h_kernel(kernel_vol);
    std::vector<float> h_output_cpu(vol_size);

    for(size_t i = 0; i < vol_size; ++i) 
        h_input[i] = (float)rand() / RAND_MAX;

    for(size_t i = 0; i < kernel_vol; ++i) 
        h_kernel[i] = (float)rand() / RAND_MAX;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timed runs
    std::vector<float> timings;
    for (int i = 0; i < warmup_iterations; ++i) {
        std::cout << "Warmup iteration " << (i+1) << "/" << warmup_iterations << std::endl;
        CUDA_CHECK(cudaEventRecord(start));
        convolution3D_gold(h_output_cpu.data(), h_input.data(), h_kernel.data(), width, height, depth, kernel_radius);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        timings.push_back(ms);
        std::cout << "  Time: " << ms << " ms" << std::endl;
    }
    print_stats("Separable + Z-Transpose", timings);
    timings.clear();
    
    return 0;
}