#include "SigmoidActivationLayer.h"
#include <cmath>


SigmoidActivationLayer::SigmoidActivationLayer(void)
{
}

ThreeDimensionalArray* SigmoidActivationLayer::forward(ThreeDimensionalArray* input){
	for(int i = 0; i < input->height * input->depth * input->width; ++i){
		input->data[i] = 0.5 * tanh(0.5 * input->data[i]) + 0.5;
	}
	return input;
}


SigmoidActivationLayer::~SigmoidActivationLayer(void)
{
}
