#include "TanhActivationLayer.h"
#include <cmath>

TanhActivationLayer::TanhActivationLayer(void)
{
}

ThreeDimensionalArray* TanhActivationLayer::forward(ThreeDimensionalArray* input){
	for(int i = 0; i < input->height * input->depth * input->width; ++i){
		input->data[i] = tanh(input->data[i]);
	}
	return input;
}

TanhActivationLayer::~TanhActivationLayer(void)
{
}
