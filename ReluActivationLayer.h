#pragma once

#include "Layer.h"

class ReluActivationLayer : public Layer
{
public:
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	ReluActivationLayer(void);
	~ReluActivationLayer(void);
};

