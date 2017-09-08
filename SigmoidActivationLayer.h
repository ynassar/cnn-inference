#pragma once

#include "Layer.h"

class SigmoidActivationLayer :
	public Layer
{
public:
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	SigmoidActivationLayer(void);
	~SigmoidActivationLayer(void);
};

