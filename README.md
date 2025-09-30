# 3D Convolution CUDA Benchmark

## Overview

This project compares several 3D convolution implementations:

* A CPU reference implementation (`convolution3D_gold`)
* A naive CUDA kernel that caches the convolution kernel in constant memory
* A naive CUDA kernel that reads the kernel from global memory
* A separable CUDA implementation

Benchmark statistics (average, median, minimum) are reported after multiple warm-up and measured iterations.

## Code organization

| Path | Description |
| --- | --- |
| `src/` | CUDA and C++ sources for the benchmark executable and helper routines. |
| `tests/` | GoogleTest-based unit tests that validate the CPU and CUDA kernels. |
| `bin/` | Build output directory containing compiled binaries such as `convolution3D`. |

## Building

The project uses a standard CUDA-enabled makefile.

```bash
make
```

The resulting binaries are placed in `bin/`.

To clean the build artifacts:

```bash
make clean
```

## Running benchmarks

The main benchmark binary is `bin/convolution3D`. Run it directly to use the default configuration:

```bash
./bin/convolution3D
```

Command-line options let you tailor the benchmark to your dataset or GPU. Run `--help` to view the supported parameters:

```bash
./bin/convolution3D --help
```

### Command-line options

| Option | Description | Default |
| --- | --- | --- |
| `--width <int>` | Volume width | 128 |
| `--height <int>` | Volume height | 128 |
| `--depth <int>` | Volume depth | 8 |
| `--kernel-radius <int>` | Convolution kernel radius (kernel size = `2 * radius + 1`) | 7 |
| `--iterations <int>` | Number of timed iterations | 20 |
| `--warmup-iterations <int>` | Number of warm-up iterations executed before timing | 5 |
| `-h, --help` | Show usage information | — |

Example: benchmark a 256×256×32 volume with a radius-5 kernel, measuring 50 iterations after 10 warm-up runs:

```bash
./bin/convolution3D --width 256 --height 256 --depth 32 \
  --kernel-radius 5 --iterations 50 --warmup-iterations 10
```

> **Note:** The kernel must fit in CUDA constant memory. Extremely large radii will be rejected with a descriptive error.


```bash
- Copy code
make clean
```

This will remove all files in the bin/ directory.


## Tests

Unit tests for the CPU and CUDA kernels live under `tests/` and require GoogleTest. Build and run them with:

```bash
- Copy code
make test
```

GoogleTest can be build with CMake using following commands:

```bash
git clone https://github.com/google/googletest.git
cd googletest/
mkdir build 
cd build
cmake ..
make
```