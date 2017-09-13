#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include "Mat.h"
#include "Mat.cpp"

namespace CNNInference{

	class FullyConnectedLayer : public Layer
	{
	public:
		CNNInference::Matrix<float>* output;
		CNNInference::Matrix<float>* weights;
		CNNInference::Matrix<float>* biases;
		CNNInference::Matrix<float>* x;
		FullyConnectedLayer(void);
		FullyConnectedLayer(int input_dimension, int output_dimension, CNNInference::Matrix<float>* weights, CNNInference::Matrix<float>* biases);
		CNNInference::Matrix<float>* forward(CNNInference::Matrix<float>* input);
		~FullyConnectedLayer(void);
	};

}
