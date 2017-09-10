#include "NaiveConvolutor.h"
#include <iostream>

void NaiveConvolutor::convolute_and_add(Matrix* image, Matrix* filter, Matrix* matrix_to_add_to, int stride){

	int output_height = matrix_to_add_to->height;
	int output_width = matrix_to_add_to->width;
	for (int i = 0; i < output_height; ++i){
		for (int j = 0; j < output_width; ++j){
			for (int k = 0; k < filter->height; ++k){
				for(int l = 0; l < filter->width; ++l){
					float a = image->element_at(i*stride+k,j*stride+l);
					float b = filter->element_at(k, l);
					matrix_to_add_to->element_at(i,j) += a * b;
				}
			}
		}
	}
}

NaiveConvolutor::NaiveConvolutor(void)
{
}


NaiveConvolutor::~NaiveConvolutor(void)
{
}
