#include "ReluActivationLayer.h"
#include <algorithm>
#include "Matrix.h"
#include "Matrix.cpp"
#include <iostream>

namespace CNNInference {
	Matrix<float>* ReluActivationLayer::forward(Matrix<float>* input) {
		int limit = input->phy_height * input->phy_width;
		for (int i = 0; i < limit; i++) {
			input->matrix[i] = std::max(input->matrix[i], 0.f);
		}
		return input;
	}

	ReluActivationLayer::ReluActivationLayer(void)
	{
	}


	ReluActivationLayer::~ReluActivationLayer(void)
	{
	}
}