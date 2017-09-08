#include "FullyConnectedLayer.h"
#include "NaiveMatrixMultiplier.h"

FullyConnectedLayer::FullyConnectedLayer(void)
{
}

FullyConnectedLayer::FullyConnectedLayer(int input_dimension, int output_dimension){
	this->weights = new Matrix(input_dimension, output_dimension);
	this->biases = new Vector(output_dimension);
}

ThreeDimensionalArray* FullyConnectedLayer::forward(ThreeDimensionalArray* input){
	Vector* input_vector = input->to_matrix()->flatten();
	Vector* output_vector = NaiveMatrixMultiplier::vector_matrix_multiply(input_vector, this->weights);

	for (int i = 0; i < this->biases->size; ++i){
		output_vector->element_at(i) += this->biases->element_at(i);
	}

	return new ThreeDimensionalArray(output_vector);
}

FullyConnectedLayer::~FullyConnectedLayer(void)
{
}
