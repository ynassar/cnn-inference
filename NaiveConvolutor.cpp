#include "NaiveConvolutor.h"

void NaiveConvolutor::convolute_and_add(Matrix* image, Matrix* filter, Matrix* matrix_to_add_to){
	for (int i = 0; i < image->height - filter->height; ++i){
		for (int j = 0; j < image->width - filter->width; ++j){
			for (int k = 0; k < filter->height; ++k){
				for(int l = 0; l < filter->width; ++l){
					matrix_to_add_to->element_at(i,j) += image->element_at(i+k,j+l) * filter->element_at(k, l); 
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
