#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <string>
#include <cstdlib>
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

struct BenchmarkConfig {
    int width = 128;
    int height = 128;
    int depth = 8;
    int kernel_radius = 7;
    int num_iterations = 20;
    int warmup_iterations = 5;
};

enum class ArgParseStatus {
    Success,
    ShowHelp,
    Error
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --width <int>             Volume width (default: 128)\n"
              << "  --height <int>            Volume height (default: 128)\n"
              << "  --depth <int>             Volume depth (default: 8)\n"
              << "  --kernel-radius <int>     Kernel radius (default: 7)\n"
              << "  --iterations <int>        Benchmark iterations (default: 20)\n"
              << "  --warmup-iterations <int> Warmup iterations (default: 5)\n"
              << "  -h, --help                Show this help message\n";
}

bool assign_positive_int(const std::string& value,
                         const std::string& name,
                         int min_value,
                         BenchmarkConfig& config,
                         int BenchmarkConfig::*member) {
    try {
        size_t processed = 0;
        const int parsed = std::stoi(value, &processed);
        if (processed != value.size() || parsed < min_value) {
            std::cerr << "Invalid value for " << name << ": " << value
                      << " (expected integer >= " << min_value << ")" << std::endl;
            return false;
        }
        config.*member = parsed;
        return true;
    } catch (const std::exception&) {
        std::cerr << "Invalid value for " << name << ": " << value << std::endl;
        return false;
    }
}

ArgParseStatus parse_arguments(int argc, char** argv, BenchmarkConfig& config) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const std::string& option_name, std::string& out_value) -> bool {
            if (i + 1 >= argc) {
                std::cerr << option_name << " requires a value" << std::endl;
                return false;
            }
            out_value = argv[++i];
            return true;
        };

        if (arg == "--width") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 1, config, &BenchmarkConfig::width)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--height") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 1, config, &BenchmarkConfig::height)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--depth") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 1, config, &BenchmarkConfig::depth)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--kernel-radius") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 0, config, &BenchmarkConfig::kernel_radius)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--iterations") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 1, config, &BenchmarkConfig::num_iterations)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--warmup-iterations") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 0, config, &BenchmarkConfig::warmup_iterations)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return ArgParseStatus::ShowHelp;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return ArgParseStatus::Error;
        }
    }

    if (config.warmup_iterations > config.num_iterations) {
        std::cerr << "Warmup iterations must not exceed total iterations" << std::endl;
        return ArgParseStatus::Error;
    }

    return ArgParseStatus::Success;
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

int main(int argc, char** argv) {
    BenchmarkConfig config;
    const ArgParseStatus arg_status = parse_arguments(argc, argv, config);
    if (arg_status == ArgParseStatus::ShowHelp) {
        return EXIT_SUCCESS;
    }
    if (arg_status == ArgParseStatus::Error) {
        return EXIT_FAILURE;
    }

    // --- Benchmark Configuration ---
    const int width = config.width;
    const int height = config.height;
    const int depth = config.depth;
    const int kernel_radius = config.kernel_radius;
    const int num_iterations = config.num_iterations;
    const int warmup_iterations = config.warmup_iterations;

    const size_t vol_size = static_cast<size_t>(width) * height * depth;
    const size_t vol_bytes = vol_size * sizeof(float);
    
    const int kernel_size = 2 * kernel_radius + 1;
    const size_t kernel_vol = static_cast<size_t>(kernel_size) * kernel_size * kernel_size;
    const size_t kernel_bytes = kernel_vol * sizeof(float);

    if (kernel_vol > static_cast<size_t>(MAX_KERNEL_CONSTANT_ELEMENTS)) {
        std::cerr << "Kernel size " << kernel_size << " (" << kernel_vol
                  << " elements) exceeds constant memory capacity ("
                  << MAX_KERNEL_CONSTANT_ELEMENTS << " elements)." << std::endl;
        return EXIT_FAILURE;
    }

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