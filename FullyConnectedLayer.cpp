#include "FullyConnectedLayer.h"
#include "Matrix.h"
#include "Matrix.cpp"
#include <iostream>

namespace CNNInference {

	FullyConnectedLayer::FullyConnectedLayer(void)
	{
	}

	FullyConnectedLayer::FullyConnectedLayer(int input_dimension, int output_dimension, Matrix<float>* weights, Matrix<float>* biases) {
		this->weights = weights;
		this->biases = biases;
		this->output = new Matrix<float>(1, output_dimension, 8);
	}

	Matrix<float>* FullyConnectedLayer::forward(Matrix<float>* input) {
		Matrix<float>* flat_input = new Matrix<float>(1, input->height * input->width, 8);
		for (int i = 0; i < input->height; i++) {
			for (int j = 0; j < input->width; j++) {
				(*flat_input)[0][i * input->width + j] = (*input)[i][j];
			}
		}
		flat_input->mult3(*this->weights, this->output);
		this->output->element_wise_add_AVX(*this->biases);
		return output;
	}

	FullyConnectedLayer::~FullyConnectedLayer(void)
	{
	}
}