#pragma once

#include "ThreeDimensionalArray.h"

class Layer
{
public:
	Layer(void);
	virtual ThreeDimensionalArray* forward(ThreeDimensionalArray* input) = 0;
	~Layer(void);
};

