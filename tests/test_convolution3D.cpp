#include <gtest/gtest.h>

#include <vector>

#include "../src/convolution3D_common.h"

namespace {

TEST(Convolution3DGoldTest, BoxFilterAllOnesVolume) {
	const int width = 3;
	const int height = 3;
	const int depth = 3;
	const int kernel_radius = 1;
	const int kernel_size = 2 * kernel_radius + 1;
	const int volume_elements = width * height * depth;
	const int kernel_elements = kernel_size * kernel_size * kernel_size;

	std::vector<float> input(volume_elements, 1.0f);
	std::vector<float> kernel(kernel_elements, 1.0f);
	std::vector<float> output(volume_elements, 0.0f);
	std::vector<float> expected(volume_elements, static_cast<float>(kernel_elements));

	convolution3D_gold(output.data(), input.data(), kernel.data(),
					   width, height, depth, kernel_radius, false);

	for (int idx = 0; idx < volume_elements; ++idx) {
		EXPECT_FLOAT_EQ(expected[idx], output[idx])
			<< "Mismatch at index " << idx;
	}
}

TEST(Convolution3DGoldTest, LaplaceFilterImpulseVolume) {
	const int width = 3;
	const int height = 3;
	const int depth = 3;
	const int kernel_radius = 1;
	const int kernel_size = 2 * kernel_radius + 1;
	const int volume_elements = width * height * depth;
	const int kernel_elements = kernel_size * kernel_size * kernel_size;

	std::vector<float> input(volume_elements, 0.0f);
	const int center_idx = (depth / 2) * (width * height) + (height / 2) * width + (width / 2);
	input[center_idx] = 1.0f;

	std::vector<float> kernel(kernel_elements, 0.0f);
	const int center_kernel_idx = (kernel_radius) * (kernel_size * kernel_size) +
								  (kernel_radius) * kernel_size + kernel_radius;
	kernel[center_kernel_idx] = -6.0f;

	auto set_neighbor = [&](int dx, int dy, int dz, float value) {
		int x = kernel_radius + dx;
		int y = kernel_radius + dy;
		int z = kernel_radius + dz;
		int idx = z * (kernel_size * kernel_size) + y * kernel_size + x;
		kernel[idx] = value;
	};

	set_neighbor(-1, 0, 0, 1.0f);
	set_neighbor(1, 0, 0, 1.0f);
	set_neighbor(0, -1, 0, 1.0f);
	set_neighbor(0, 1, 0, 1.0f);
	set_neighbor(0, 0, -1, 1.0f);
	set_neighbor(0, 0, 1, 1.0f);

	std::vector<float> output(volume_elements, 0.0f);
	std::vector<float> expected(volume_elements, 0.0f);

	for (int z = 0; z < depth; ++z) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				const int output_idx = z * (width * height) + y * width + x;
				const int kernel_x = (kernel_size - 1) - x;
				const int kernel_y = (kernel_size - 1) - y;
				const int kernel_z = (kernel_size - 1) - z;
				const int kernel_idx = kernel_z * (kernel_size * kernel_size) +
									   kernel_y * kernel_size + kernel_x;
				expected[output_idx] = kernel[kernel_idx];
			}
		}
	}

	convolution3D_gold(output.data(), input.data(), kernel.data(),
					   width, height, depth, kernel_radius, false);

	for (int idx = 0; idx < volume_elements; ++idx) {
		EXPECT_FLOAT_EQ(expected[idx], output[idx])
			<< "Mismatch at index " << idx;
	}
}

TEST(Convolution3DGoldTest, BoxFilterAllOnesVolumeZeroPadding) {
	const int width = 3;
	const int height = 3;
	const int depth = 3;
	const int kernel_radius = 1;
	const int kernel_size = 2 * kernel_radius + 1;
	const int volume_elements = width * height * depth;
	const int kernel_elements = kernel_size * kernel_size * kernel_size;

	std::vector<float> input(volume_elements, 1.0f);
	std::vector<float> kernel(kernel_elements, 1.0f);
	std::vector<float> output(volume_elements, 0.0f);
	std::vector<float> expected(volume_elements, 0.0f);

	// For zero padding, corners have 8 contributions (2x2x2), edges 18 (2x3x3 or 3x2x3 etc), faces 18, center 27
	for (int z = 0; z < depth; ++z) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int idx = z * (width * height) + y * width + x;
				int num_contrib = (x > 0 ? (x < width-1 ? 3 : 2) : 2) *
								  (y > 0 ? (y < height-1 ? 3 : 2) : 2) *
								  (z > 0 ? (z < depth-1 ? 3 : 2) : 2);
				expected[idx] = static_cast<float>(num_contrib);
			}
		}
	}

	convolution3D_gold(output.data(), input.data(), kernel.data(),
					   width, height, depth, kernel_radius, true);

	for (int idx = 0; idx < volume_elements; ++idx) {
		EXPECT_FLOAT_EQ(expected[idx], output[idx])
			<< "Mismatch at index " << idx;
	}
}

}  // namespace
