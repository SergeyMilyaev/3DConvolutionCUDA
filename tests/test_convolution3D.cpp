#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>

#include "../src/convolution3D_common.h"

namespace {

class Convolution3DTest : public ::testing::Test {
protected:
	const int width = 3;
	const int height = 3;
	const int depth = 3;
	const int kernel_radius = 1;
	const int kernel_size = 2 * kernel_radius + 1;
	const int volume_elements = width * height * depth;
	const int kernel_elements = kernel_size * kernel_size * kernel_size;

	std::vector<float> input;
	std::vector<float> kernel;
	std::vector<float> output;

	void SetUp() override {
		input.resize(volume_elements);
		kernel.resize(kernel_elements);
		output.resize(volume_elements);
	}

	void setupBoxFilterKernel() {
		std::fill(kernel.begin(), kernel.end(), 1.0f);
	}

	void setupOnesVolume() {
		std::fill(input.begin(), input.end(), 1.0f);
	}

	void setupImpulseVolume() {
		std::fill(input.begin(), input.end(), 0.0f);
		const int center_idx = (depth / 2) * (width * height) + (height / 2) * width + (width / 2);
		input[center_idx] = 1.0f;
	}

	void setupLaplaceKernel() {
		std::fill(kernel.begin(), kernel.end(), 0.0f);
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
	}
};

class Convolution3DGoldTest : public Convolution3DTest {};

TEST_F(Convolution3DGoldTest, BoxFilterAllOnesVolume) {
	setupOnesVolume();
	setupBoxFilterKernel();
	std::vector<float> expected(volume_elements, static_cast<float>(kernel_elements));

	convolution3D_gold(output.data(), input.data(), kernel.data(),
					   width, height, depth, kernel_radius, kernel_radius, kernel_radius, false);

	for (int idx = 0; idx < volume_elements; ++idx) {
		EXPECT_FLOAT_EQ(expected[idx], output[idx])
			<< "Mismatch at index " << idx;
	}
}

TEST_F(Convolution3DGoldTest, LaplaceFilterImpulseVolume) {
	setupImpulseVolume();
	setupLaplaceKernel();

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
					   width, height, depth, kernel_radius, kernel_radius, kernel_radius, false);

	for (int idx = 0; idx < volume_elements; ++idx) {
		EXPECT_FLOAT_EQ(expected[idx], output[idx])
			<< "Mismatch at index " << idx;
	}
}

TEST_F(Convolution3DGoldTest, BoxFilterAllOnesVolumeZeroPadding) {
	setupOnesVolume();
	setupBoxFilterKernel();
	std::vector<float> expected(volume_elements, 0.0f);

	// For zero padding, corners have 8 contributions (2x2x2), edges 18 (2x3x3 or 3x2x3 etc), faces 18, center 27
	for (int z = 0; z < depth; ++z) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int idx = z * (width * height) + y * width + x;
				int contr_x_max = 3;
				int contr_y_max = 3;
				int contr_z_max = 3;
				int num_contrib = (x > 0 ? (x < width-1 ? contr_x_max : --contr_x_max) : --contr_x_max) *
								  (y > 0 ? (y < height-1 ? contr_y_max : --contr_y_max) : --contr_y_max) *
								  (z > 0 ? (z < depth-1 ? contr_z_max : --contr_z_max) : --contr_z_max);
				expected[idx] = static_cast<float>(num_contrib);
			}
		}
	}

	convolution3D_gold(output.data(), input.data(), kernel.data(),
					   width, height, depth, kernel_radius, kernel_radius, kernel_radius, true);

	for (int idx = 0; idx < volume_elements; ++idx) {
		EXPECT_FLOAT_EQ(expected[idx], output[idx])
			<< "Mismatch at index " << idx;
	}
}

class Convolution3DCUDATest : public Convolution3DTest {
protected:
	float *d_input = nullptr, *d_output = nullptr;

	void SetUp() override {
		Convolution3DTest::SetUp();
		cudaMalloc(&d_input, volume_elements * sizeof(float));
		cudaMalloc(&d_output, volume_elements * sizeof(float));
	}

	void TearDown() override {
		cudaFree(d_input);
		cudaFree(d_output);
	}

	void runCUDATest(bool use_zero_padding) {
		std::vector<float> expected(volume_elements);
		convolution3D_gold(expected.data(), input.data(), kernel.data(),
						   width, height, depth, kernel_radius, kernel_radius, kernel_radius, use_zero_padding);

		cudaMemcpy(d_input, input.data(), volume_elements * sizeof(float), cudaMemcpyHostToDevice);

		cudaError_t err = uploadConvolutionKernelToConstantMemory(kernel.data(), kernel_radius, kernel_radius, kernel_radius);
		ASSERT_EQ(err, cudaSuccess);

		dim3 blockDim(2, 2, 2);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
					 (height + blockDim.y - 1) / blockDim.y,
					 (depth + blockDim.z - 1) / blockDim.z);
		size_t shared_size = convolution3DSharedTileSizeBytes(blockDim, kernel_radius, kernel_radius, kernel_radius);
		launch_convolution3DOptimized(gridDim, blockDim, shared_size, d_output, d_input, width, height, depth, kernel_radius, kernel_radius, kernel_radius, use_zero_padding);

		err = cudaDeviceSynchronize();
		ASSERT_EQ(err, cudaSuccess);

		cudaMemcpy(output.data(), d_output, volume_elements * sizeof(float), cudaMemcpyDeviceToHost);

		for (int idx = 0; idx < volume_elements; ++idx) {
			EXPECT_FLOAT_EQ(expected[idx], output[idx])
				<< "Mismatch at index " << idx;
		}
	}
};

TEST_F(Convolution3DCUDATest, BoxFilterAllOnesVolumeZeroPadding) {
	setupOnesVolume();
	setupBoxFilterKernel();
	runCUDATest(true);
}

TEST_F(Convolution3DCUDATest, LaplaceFilterImpulseVolumeZeroPadding) {
	setupImpulseVolume();
	setupLaplaceKernel();
	runCUDATest(true);
}

TEST_F(Convolution3DCUDATest, BoxFilterAllOnesVolumeNoPadding) {
	setupOnesVolume();
	setupBoxFilterKernel();
	runCUDATest(false);
}

class Convolution3DCUDAGlobalTest : public Convolution3DTest {
protected:
	float *d_input = nullptr, *d_output = nullptr, *d_kernel = nullptr;

	void SetUp() override {
		Convolution3DTest::SetUp();
		cudaMalloc(&d_input, volume_elements * sizeof(float));
		cudaMalloc(&d_output, volume_elements * sizeof(float));
		cudaMalloc(&d_kernel, kernel_elements * sizeof(float));
	}

	void TearDown() override {
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_kernel);
	}

	void runCUDAGlobalTest(bool use_zero_padding) {
		std::vector<float> expected(volume_elements);
		convolution3D_gold(expected.data(), input.data(), kernel.data(),
						   width, height, depth, kernel_radius, kernel_radius, kernel_radius, use_zero_padding);

		cudaMemcpy(d_input, input.data(), volume_elements * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_kernel, kernel.data(), kernel_elements * sizeof(float), cudaMemcpyHostToDevice);

		dim3 blockDim(2, 2, 2);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
					 (height + blockDim.y - 1) / blockDim.y,
					 (depth + blockDim.z - 1) / blockDim.z);
		launch_convolution3DBaseline(gridDim, blockDim, d_output, d_input, d_kernel, width, height, depth, kernel_radius, kernel_radius, kernel_radius, use_zero_padding);

		cudaError_t err = cudaDeviceSynchronize();
		ASSERT_EQ(err, cudaSuccess);

		cudaMemcpy(output.data(), d_output, volume_elements * sizeof(float), cudaMemcpyDeviceToHost);

		for (int idx = 0; idx < volume_elements; ++idx) {
			EXPECT_FLOAT_EQ(expected[idx], output[idx])
				<< "Mismatch at index " << idx;
		}
	}
};

TEST_F(Convolution3DCUDAGlobalTest, BoxFilterAllOnesVolumeZeroPadding) {
	setupOnesVolume();
	setupBoxFilterKernel();
	runCUDAGlobalTest(true);
}

TEST_F(Convolution3DCUDAGlobalTest, LaplaceFilterImpulseVolumeZeroPadding) {
	setupImpulseVolume();
	setupLaplaceKernel();
	runCUDAGlobalTest(true);
}

class Convolution3DSeparableCUDATest : public Convolution3DTest {
protected:
	float *d_input = nullptr, *d_output = nullptr;
	std::vector<float> kernel_x;
	std::vector<float> kernel_y;
	std::vector<float> kernel_z;

	void SetUp() override {
		Convolution3DTest::SetUp();
		kernel_x.resize(kernel_size);
		kernel_y.resize(kernel_size);
		kernel_z.resize(kernel_size);
		ASSERT_EQ(cudaSuccess, cudaMalloc(&d_input, volume_elements * sizeof(float)));
		ASSERT_EQ(cudaSuccess, cudaMalloc(&d_output, volume_elements * sizeof(float)));
	}

	void TearDown() override {
		cudaFree(d_input);
		cudaFree(d_output);
	}

	void setupSeparableKernel() {
		ASSERT_EQ(kernel_size, 3);
		// Symmetric kernels for each axis
		const float axis_x[] = {0.25f, 0.5f, 0.25f};
		const float axis_y[] = {0.5f, 1.0f, 0.5f};
		const float axis_z[] = {1.0f, 2.0f, 1.0f};
		for (int i = 0; i < kernel_size; ++i) {
			kernel_x[i] = axis_x[i];
			kernel_y[i] = axis_y[i];
			kernel_z[i] = axis_z[i];
		}

		for (int z = 0; z < kernel_size; ++z) {
			for (int y = 0; y < kernel_size; ++y) {
				for (int x = 0; x < kernel_size; ++x) {
					const int idx = z * (kernel_size * kernel_size) + y * kernel_size + x;
					kernel[idx] = kernel_x[x] * kernel_y[y] * kernel_z[z];
				}
			}
		}
	}

	void setupRampVolume() {
		for (int idx = 0; idx < volume_elements; ++idx) {
			input[idx] = static_cast<float>((idx % 7) - 3);
		}
	}

	void runSeparableTest(bool use_zero_padding, dim3 block_dim = dim3(0, 0, 0)) {
		std::vector<float> expected(volume_elements);
		convolution3D_gold(expected.data(), input.data(), kernel.data(),
				width, height, depth,
				kernel_radius, kernel_radius, kernel_radius,
				use_zero_padding);

		cudaError_t copy_err = cudaMemcpy(d_input, input.data(), volume_elements * sizeof(float), cudaMemcpyHostToDevice);
		ASSERT_EQ(cudaSuccess, copy_err);

		cudaError_t err = convolution3DSeparable(d_output, d_input,
			kernel_x.data(), kernel_y.data(), kernel_z.data(),
			width, height, depth,
			kernel_radius, kernel_radius, kernel_radius,
			use_zero_padding,
			nullptr,
			block_dim);
		ASSERT_EQ(err, cudaSuccess);

		err = cudaDeviceSynchronize();
		ASSERT_EQ(err, cudaSuccess);

		copy_err = cudaMemcpy(output.data(), d_output, volume_elements * sizeof(float), cudaMemcpyDeviceToHost);
		ASSERT_EQ(cudaSuccess, copy_err);

		for (int idx = 0; idx < volume_elements; ++idx) {
			EXPECT_NEAR(expected[idx], output[idx], 1e-4f)
				<< "Mismatch at index " << idx;
		}
	}
};

TEST_F(Convolution3DSeparableCUDATest, RampVolumeClampPadding) {
	setupRampVolume();
	setupSeparableKernel();
	runSeparableTest(false);
}

TEST_F(Convolution3DSeparableCUDATest, RampVolumeZeroPadding) {
	setupRampVolume();
	setupSeparableKernel();
	runSeparableTest(true);
}

TEST_F(Convolution3DSeparableCUDATest, CustomBlockConfiguration) {
	setupRampVolume();
	setupSeparableKernel();
	const dim3 custom_block(3, 2, 1);
	runSeparableTest(false, custom_block);
}

TEST_F(Convolution3DCUDAGlobalTest, BoxFilterAllOnesVolumeNoPadding) {
	setupOnesVolume();
	setupBoxFilterKernel();
	runCUDAGlobalTest(false);
}

TEST_F(Convolution3DCUDAGlobalTest, LaplaceFilterImpulseVolumeNoPadding) {
	setupImpulseVolume();
	setupLaplaceKernel();
	runCUDAGlobalTest(false);
}

}  // namespace
