#include "SigmoidActivationLayer.h"
#include <cmath>


namespace CNNInference {
	SigmoidActivationLayer::SigmoidActivationLayer(void)
	{
	}

	Matrix<float>* SigmoidActivationLayer::forward(Matrix<float>* input) {
		int limit = input->phy_height * input->phy_width;
		for (int i = 0; i < limit; i++) {
			input->matrix[i] = 0.5f * tanh(0.5f * input->matrix[i]) + 0.5f;
		}
		return input;
	}


	SigmoidActivationLayer::~SigmoidActivationLayer(void)
	{
	}
}