#include "SigmoidActivationLayer.h"
#include <cmath>


SigmoidActivationLayer::SigmoidActivationLayer(void)
{
}

ThreeDimensionalArray* SigmoidActivationLayer::forward(ThreeDimensionalArray* input){
	for(int i = 0; i < input->depth; ++i){
		Matrix* matrix_at_depth = input->matrix_at(i);
		for(int j = 0; j < matrix_at_depth->height; ++j){
			for(int k = 0; k < matrix_at_depth->width; ++k){
				matrix_at_depth->element_at(j,k) = 0.5 * tanh(0.5 * matrix_at_depth->element_at(j,k)) + 0.5;

			}
		}
	}
	return input;
}


SigmoidActivationLayer::~SigmoidActivationLayer(void)
{
}
