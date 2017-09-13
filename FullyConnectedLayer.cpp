#include "FullyConnectedLayer.h"
#include "Mat.h"
#include "Mat.cpp"
#include <iostream>
using namespace std;
using namespace CNNInference;

FullyConnectedLayer::FullyConnectedLayer(void)
{
}

FullyConnectedLayer::FullyConnectedLayer(int input_dimension, int output_dimension, CNNInference::Mat<float>* weights, CNNInference::Mat<float>* biases){
	this->weights = weights;
	this->biases = biases;
	this->output = new CNNInference::Mat<float>(1, output_dimension, 8);
}

CNNInference::Mat<float>* FullyConnectedLayer::forward(CNNInference::Mat<float>* input){
	CNNInference::Mat<float>* flat_input = new CNNInference::Mat<float>(1, input->height * input->width, 8);
	for (int i = 0; i < input->height; i ++){
		for (int j = 0; j < input->width; j ++){
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
