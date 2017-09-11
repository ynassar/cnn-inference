/*
 * PoolingLayer.cpp
 *
 *  Created on: Sep 10, 2017
 *      Author: yousefnassar
 */

#include "PoolingLayer.h"
#include <algorithm>
#include <limits>
#include <iostream>
using namespace std;

PoolingLayer::PoolingLayer() {
	// TODO Auto-generated constructor stub

}

ThreeDimensionalArray* PoolingLayer::forward(ThreeDimensionalArray* input){
	int output_height = (input->height - this->kernel_size) / this->stride + 1;
	if((input->height - this->kernel_size) % this->stride )
		output_height ++;
	int output_width = (input->width - this->kernel_size) / this->stride + 1;
	if ((input->width - this->kernel_size) % this->stride )
		output_width ++;

	ThreeDimensionalArray* output = new ThreeDimensionalArray(input->depth, output_height, output_width);
	for (int i = 0; i < input->depth; i ++){
		Matrix* matrix = input->matrix_at(i);
		Matrix* out_matrix = output->matrix_at(i);
		for (int row_start = 0; row_start < output_height; row_start ++){
			for (int col_start = 0; col_start < output_width; col_start ++){
				float mval = -std::numeric_limits<float>::max();
				for (int filter_i = 0; filter_i < this->kernel_size; filter_i ++){
					for (int filter_j = 0; filter_j < this->kernel_size; filter_j ++){
						int mat_i = row_start * this->stride + filter_i;
						int mat_j = col_start * this->stride + filter_j;
						if (mat_i < matrix->height && mat_j < matrix->width)
							mval = max(mval, matrix->element_at(mat_i, mat_j));
					}
				}
				out_matrix->element_at(row_start, col_start) = mval;
			}
		}
	}
	return output;
}

PoolingLayer::~PoolingLayer() {
	// TODO Auto-generated destructor stub
}

