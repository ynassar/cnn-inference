#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include "Matrix.h"
#include "Matrix.cpp"

namespace CNNInference{

	class FullyConnectedLayer : public Layer
	{
	public:
		Matrix<float>* output;
		Matrix<float>* weights;
		Matrix<float>* biases;
		Matrix<float>* x;
		Matrix<float>* flat_input;
		FullyConnectedLayer(void);
		FullyConnectedLayer(int input_dimension, int output_dimension, CNNInference::Matrix<float>* weights, CNNInference::Matrix<float>* biases);
		CNNInference::Matrix<float>* forward(CNNInference::Matrix<float>* input);
		~FullyConnectedLayer(void);
	};

}
