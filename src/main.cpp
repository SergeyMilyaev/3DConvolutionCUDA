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
    int kernel_radius_x = 7;
    int kernel_radius_y = 7;
    int kernel_radius_z = 7;
    int num_iterations = 20;
    int warmup_iterations = 5;
    int block_dim_x = 8;
    int block_dim_y = 8;
    int block_dim_z = 8;
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
              << "  --kernel-radius <int>     Kernel radius for all axes (default: 7)\n"
              << "  --kernel-radius-x <int>   Kernel radius along X axis (default: 7)\n"
              << "  --kernel-radius-y <int>   Kernel radius along Y axis (default: 7)\n"
              << "  --kernel-radius-z <int>   Kernel radius along Z axis (default: 7)\n"
              << "  --iterations <int>        Benchmark iterations (default: 20)\n"
              << "  --warmup-iterations <int> Warmup iterations (default: 5)\n"
              << "  --block-dim-x <int>       CUDA block dimension X (default: 8)\n"
              << "  --block-dim-y <int>       CUDA block dimension Y (default: 8)\n"
              << "  --block-dim-z <int>       CUDA block dimension Z (default: 8)\n"
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
        } else if (arg == "--kernel-radius-x") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 0, config, &BenchmarkConfig::kernel_radius_x)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--kernel-radius-y") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 0, config, &BenchmarkConfig::kernel_radius_y)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--kernel-radius-z") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 0, config, &BenchmarkConfig::kernel_radius_z)) {
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
        } else if (arg == "--block-dim-x") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 1, config, &BenchmarkConfig::block_dim_x)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--block-dim-y") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 1, config, &BenchmarkConfig::block_dim_y)) {
                return ArgParseStatus::Error;
            }
        } else if (arg == "--block-dim-z") {
            std::string value;
            if (!require_value(arg, value)) return ArgParseStatus::Error;
            if (!assign_positive_int(value, arg, 1, config, &BenchmarkConfig::block_dim_z)) {
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
    const int kernel_radius_x = config.kernel_radius_x;
    const int kernel_radius_y = config.kernel_radius_y;
    const int kernel_radius_z = config.kernel_radius_z;
    const int num_iterations = config.num_iterations;
    const int warmup_iterations = config.warmup_iterations;
    const int block_dim_x = config.block_dim_x;
    const int block_dim_y = config.block_dim_y;
    const int block_dim_z = config.block_dim_z;

    const size_t vol_size = static_cast<size_t>(width) * height * depth;
    const size_t vol_bytes = vol_size * sizeof(float);

    const int kernel_dim_x = 2 * kernel_radius_x + 1;
    const int kernel_dim_y = 2 * kernel_radius_y + 1;
    const int kernel_dim_z = 2 * kernel_radius_z + 1;
    const size_t kernel_volume = static_cast<size_t>(kernel_dim_x) * kernel_dim_y * kernel_dim_z;
    const size_t kernel_bytes = kernel_volume * sizeof(float);

    if (kernel_volume > static_cast<size_t>(MAX_KERNEL_CONSTANT_ELEMENTS)) {
        std::cerr << "Kernel size "
                  << kernel_dim_x << "x" << kernel_dim_y << "x" << kernel_dim_z
                  << " (" << kernel_volume
                  << " elements) exceeds constant memory capacity ("
                  << MAX_KERNEL_CONSTANT_ELEMENTS << " elements)." << std::endl;
        return EXIT_FAILURE;
    }

    if (static_cast<long long>(block_dim_x) * block_dim_y * block_dim_z > 1024LL) {
        std::cerr << "Block dimensions produce "
                  << static_cast<long long>(block_dim_x) * block_dim_y * block_dim_z
                  << " threads per block, exceeding the CUDA limit of 1024." << std::endl;
        return EXIT_FAILURE;
    }
    if (block_dim_x > 1024 || block_dim_y > 1024 || block_dim_z > 64) {
        std::cerr << "Block dimensions ("
                  << block_dim_x << ", " << block_dim_y << ", " << block_dim_z
                  << ") exceed typical CUDA per-dimension limits (1024, 1024, 64)." << std::endl;
        return EXIT_FAILURE;
    }

    // --- Host Data Allocation and Initialization ---
    std::vector<float> host_input(vol_size);
    std::vector<float> host_kernel(kernel_volume);
    std::vector<float> host_kernel_x(kernel_dim_x);
    std::vector<float> host_kernel_y(kernel_dim_y);
    std::vector<float> host_kernel_z(kernel_dim_z);
    std::vector<float> host_output_cpu(vol_size);
    std::vector<float> host_output_naive(vol_size);
    std::vector<float> host_output_naive_global(vol_size);
    std::vector<float> host_output_separable(vol_size);

    for(size_t i = 0; i < vol_size; ++i) 
        host_input[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < kernel_dim_x; ++i) {
        host_kernel_x[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < kernel_dim_y; ++i) {
        host_kernel_y[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < kernel_dim_z; ++i) {
        host_kernel_z[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int kz = 0; kz < kernel_dim_z; ++kz) {
        for (int ky = 0; ky < kernel_dim_y; ++ky) {
            for (int kx = 0; kx < kernel_dim_x; ++kx) {
                const int idx = kz * kernel_dim_x * kernel_dim_y + ky * kernel_dim_x + kx;
                host_kernel[idx] = host_kernel_x[kx] * host_kernel_y[ky] * host_kernel_z[kz];
            }
        }
    }

    auto measure_cpu_iteration = [&]() -> float {
        const auto start = std::chrono::steady_clock::now();
        convolution3DGold(host_output_cpu.data(), host_input.data(), host_kernel.data(),
                           width, height, depth,
                           kernel_radius_x, kernel_radius_y, kernel_radius_z,
                           false);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<float, std::milli> elapsed = end - start;
        return elapsed.count();
    };

    std::cout << "=== CPU Reference (convolution3DGold) ===" << std::endl;
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
    print_stats("CPU convolution3DGold", cpu_timings);

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
    float* device_input = nullptr;
    float* d_output = nullptr;
    float* d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), vol_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, host_kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(uploadConvolutionKernelToConstantMemory(
        host_kernel.data(), kernel_radius_x, kernel_radius_y, kernel_radius_z));

    const dim3 block_dim(block_dim_x, block_dim_y, block_dim_z);
    const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                        (height + block_dim.y - 1) / block_dim.y,
                        (depth + block_dim.z - 1) / block_dim.z);
    const size_t shared_size = convolution3DSharedTileSizeBytes(
        block_dim, kernel_radius_x, kernel_radius_y, kernel_radius_z);

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
                  << max_abs_difference(host_output_cpu, host_output)
                  << std::endl;
    };

    const std::function<void(void)> launch_naive = [&]() {
        launchConvolution3DOptimized(grid_dim, block_dim, shared_size,
                                   d_output, device_input,
                                   width, height, depth,
                                   kernel_radius_x, kernel_radius_y, kernel_radius_z,
                                   false);
    };

    const std::function<void(void)> launch_naive_global = [&]() {
        launchConvolution3DBaseline(grid_dim, block_dim,
                                          d_output, device_input, d_kernel,
                                          width, height, depth,
                                          kernel_radius_x, kernel_radius_y, kernel_radius_z,
                                          false);
    };

    const std::function<void(void)> launch_separable = [&]() {
        CUDA_CHECK(convolution3DSeparable(d_output, device_input,
                                          host_kernel_x.data(), host_kernel_y.data(), host_kernel_z.data(),
                                          width, height, depth,
                                          kernel_radius_x, kernel_radius_y, kernel_radius_z,
                                          false,
                                          nullptr,
                                          block_dim));
    };

    std::vector<float> timings_naive;
    benchmark_cuda_kernel("CUDA convolution3DOptimized",
                          launch_naive,
                          timings_naive,
                          host_output_naive);

    std::vector<float> timings_naive_global;
    benchmark_cuda_kernel("CUDA convolution3DBaseline",
                          launch_naive_global,
                          timings_naive_global,
                          host_output_naive_global);

    std::vector<float> timings_separable;
    benchmark_cuda_kernel("CUDA convolution3DSeparable",
                          launch_separable,
                          timings_separable,
                          host_output_separable);

    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
    
    return 0;
}