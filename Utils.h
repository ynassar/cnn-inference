#pragma once

#include "Matrix.h"
#include <string>

namespace CNNInference {

	Matrix<float>* matrix_from_file(const std::string& txtfile);
	void tile(float* input, float* tiled, int in_channels, int in_height, int in_width, int tile_size);
}