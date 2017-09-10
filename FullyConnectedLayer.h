#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"

class FullyConnectedLayer : public Layer
{
public:
	Matrix* weights;
	Vector* biases;
	FullyConnectedLayer(void);
	FullyConnectedLayer(int input_dimension, int output_dimension);
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	~FullyConnectedLayer(void);
};

