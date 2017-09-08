#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"

class FullyConnectedLayer : Layer
{
	Matrix* weights;
	Vector* biases;
public:
	FullyConnectedLayer(void);
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	~FullyConnectedLayer(void);
};

