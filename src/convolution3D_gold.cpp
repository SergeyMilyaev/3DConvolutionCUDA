#include <cmath>
#include <algorithm>

// Helper function to clamp coordinates to the volume boundaries
int clamp(int x, int a, int b) {
    return std::max(a, std::min(x, b - 1));
}

// Host-side reference implementation of 3D convolution
void convolution3D_gold(
    float* p_output,
    const float* p_input,
    const float* p_kernel,
    const int width,
    const int height,
    const int depth,
    const int kernel_radius,
    const bool use_zero_padding) 
{
    const int kernel_size = 2 * kernel_radius + 1;
    const int kernel_volume = kernel_size * kernel_size * kernel_size;

    // Iterate over each voxel in the output volume
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                
                float sum = 0.0f;

                // Iterate over the kernel
                for (int kz = 0; kz < kernel_size; ++kz) {
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            
                            if (use_zero_padding) {
                                // Calculate input coordinates
                                int ix = x + kx - kernel_radius;
                                int iy = y + ky - kernel_radius;
                                int iz = z + kz - kernel_radius;

                                // Only add if within bounds
                                if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 && iz < depth) {
                                    int input_idx = iz * (width * height) + iy * width + ix;
                                    int kernel_idx = kz * (kernel_size * kernel_size) + ky * kernel_size + kx;
                                    sum += p_input[input_idx] * p_kernel[kernel_idx];
                                }
                            } else {
                                // Calculate input coordinates with clamping
                                int ix = clamp(x + kx - kernel_radius, 0, width);
                                int iy = clamp(y + ky - kernel_radius, 0, height);
                                int iz = clamp(z + kz - kernel_radius, 0, depth);

                                // Linearize indices
                                int input_idx = iz * (width * height) + iy * width + ix;
                                int kernel_idx = kz * (kernel_size * kernel_size) + ky * kernel_size + kx;

                                sum += p_input[input_idx] * p_kernel[kernel_idx];
                            }
                        }
                    }
                }
                
                // Write the result to the output volume
                int output_idx = z * (width * height) + y * width + x;
                p_output[output_idx] = sum;
            }
        }
    }
}