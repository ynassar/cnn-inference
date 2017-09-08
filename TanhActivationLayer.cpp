#include "TanhActivationLayer.h"
#include <cmath>

TanhActivationLayer::TanhActivationLayer(void)
{
}

ThreeDimensionalArray* TanhActivationLayer::forward(ThreeDimensionalArray* input){
	for(int i = 0; i < input->depth; ++i){
		Matrix* matrix_at_depth = input->matrix_at(i);
		for(int j = 0; j < matrix_at_depth->height; ++j){
			for(int k = 0; k < matrix_at_depth->width; ++k){
				matrix_at_depth->element_at(j,k) = tanh(matrix_at_depth->element_at(j,k));
			}
		}
	}
	return input;
}

TanhActivationLayer::~TanhActivationLayer(void)
{
}
