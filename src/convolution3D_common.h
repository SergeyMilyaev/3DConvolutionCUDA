#ifndef CONVOLUTION3D_COMMON_H
#define CONVOLUTION3D_COMMON_H

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
void convolution3D_gold(float* p_output, const float* p_input, const float* p_kernel,
    const int width, const int height, const int depth, const int kernel_radius, const bool use_zero_padding);

#endif