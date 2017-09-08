#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"

class ConvolutionalLayer : Layer
{
	ThreeDimensionalArray* filters;
	int input_depth;
	int output_depth;
	int filter_size;
	int stride;
	int padding;
public:
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	ConvolutionalLayer(void);
	~ConvolutionalLayer(void);
};

