#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include "Mat.h"
#include "Mat.cpp"

namespace CNNInference{

	class FullyConnectedLayer : public Layer
	{
	public:
		CNNInference::Mat<float>* output;
		CNNInference::Mat<float>* weights;
		CNNInference::Mat<float>* biases;
		CNNInference::Mat<float>* x;
		FullyConnectedLayer(void);
		FullyConnectedLayer(int input_dimension, int output_dimension, CNNInference::Mat<float>* weights, CNNInference::Mat<float>* biases);
		CNNInference::Mat<float>* forward(CNNInference::Mat<float>* input);
		~FullyConnectedLayer(void);
	};

}
