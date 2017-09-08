#include "ReluActivationLayer.h"
#include <algorithm>
using namespace std;

ThreeDimensionalArray* ReluActivationLayer::forward(ThreeDimensionalArray* input){
	for(int i = 0; i < input->height * input->depth * input->width; ++i){
		input->data[i] = max(input->data[i], 0.f);
	}
	return input;
}

ReluActivationLayer::ReluActivationLayer(void)
{
}


ReluActivationLayer::~ReluActivationLayer(void)
{
}
