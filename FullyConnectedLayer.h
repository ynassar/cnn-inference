#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"

class FullyConnectedLayer : public Layer
{
	Matrix* weights;
	Vector* biases;
public:
	FullyConnectedLayer(void);
	FullyConnectedLayer(int input_dimension, int output_dimension);
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	~FullyConnectedLayer(void);
};

