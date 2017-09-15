#pragma once

#include "Matrix.h"
#include <string>

namespace CNNInference {

	Matrix<float>* matrix_from_file(const std::string& txtfile);
	void tile_and_transform_2_3(float* image, float* out_transformed, int in_channels, int in_height, int in_width);
	void transform_kernels_2_3(float* kernels, float* output, int out_channels, int in_channels);
	void elementwise_mult_and_add(float* transformed_image, float* transformed_kernels, float* output, int in_channels, int out_channels, int in_height, int in_width);
	void inverse_transform(float* multiplied_input, float* output, int out_channels, int in_height, int in_width);
}