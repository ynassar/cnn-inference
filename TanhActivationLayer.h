#pragma once

#include "Layer.h"

class TanhActivationLayer :
	public Layer
{
public:
	TanhActivationLayer(void);
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	~TanhActivationLayer(void);
};

