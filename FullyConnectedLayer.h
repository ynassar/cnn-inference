#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include "Mat.h"
#include "Mat.cpp"

namespace CNNInference{

	class FullyConnectedLayer : public Layer
	{
	public:
		Utils::Mat<float>* output;
		Utils::Mat<float>* weights;
		Utils::Mat<float>* biases;
		Utils::Mat<float>* x;
		FullyConnectedLayer(void);
		FullyConnectedLayer(int input_dimension, int output_dimension, Utils::Mat<float>* weights, Utils::Mat<float>* biases);
		Utils::Mat<float>* forward(Utils::Mat<float>* input);
		~FullyConnectedLayer(void);
	};

}
