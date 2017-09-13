/*
 * PoolingLayer.cpp
 *
 *  Created on: Sep 10, 2017
 *      Author: yousefnassar
 */

#include "PoolingLayer.h"
#include <algorithm>
#include "Matrix.cpp"
#include <limits>
#include <iostream>

namespace CNNInference {

	PoolingLayer::PoolingLayer(int input_depth, int input_height, int input_width, int kernel_size, int stride) {
		this->input_depth = input_depth;
		this->input_height = input_height;
		this->input_width = input_width;
		this->kernel_size = kernel_size;
		this->stride = stride;
		this->output_height = (input_height - this->kernel_size) / this->stride + 1;
		if ((input_height - this->kernel_size) % this->stride)
			output_height++;
		this->output_width = (input_width - this->kernel_size) / this->stride + 1;
		if ((input_width - this->kernel_size) % this->stride)
			output_width++;

		this->output = new CNNInference::Matrix<float>(input_depth, this->output_height * this->output_width, 8);
	}

	Matrix<float>* PoolingLayer::forward(Matrix<float>* input) {
		for (int i = 0; i < this->input_depth; i++) {
			float* matrix = (*input)[i];
			float* out_matrix = (*this->output)[i];
			for (int row_start = 0; row_start < this->output_height; row_start++) {
				for (int col_start = 0; col_start < this->output_width; col_start++) {
					float mval = -std::numeric_limits<float>::max();
					for (int filter_i = 0; filter_i < this->kernel_size; filter_i++) {
						for (int filter_j = 0; filter_j < this->kernel_size; filter_j++) {
							int mat_i = row_start * this->stride + filter_i;
							int mat_j = col_start * this->stride + filter_j;
							if (mat_i < this->input_height && mat_j < this->input_width)
								mval = std::max(mval, matrix[this->input_width * mat_i + mat_j]);
						}
					}
					out_matrix[row_start*this->output_width + col_start] = mval;
				}
			}
		}
		return this->output;
	}

	PoolingLayer::~PoolingLayer() {
		// TODO Auto-generated destructor stub
	}

}