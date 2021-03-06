/*
 * SoftmaxActivationLayer.cpp
 *
 *  Created on: Sep 10, 2017
 *      Author: yousefnassar
 */

#include "SoftmaxActivationLayer.h"
#include <limits>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "Matrix.cpp"

namespace CNNInference {

	SoftmaxActivationLayer::SoftmaxActivationLayer() {
		// TODO Auto-generated constructor stub

	}

	Matrix<float>* SoftmaxActivationLayer::forward(Matrix<float>* input) {
		float ymax = -std::numeric_limits<float>::max();
		int total_data_size = input->height * input->width;
		for (int i = 0; i < total_data_size; i++) {
			ymax = std::max(ymax, input->matrix[i]);
		}
		float sum = 0;
		for (int i = 0; i < total_data_size; i++) {
			input->matrix[i] = exp(input->matrix[i] - ymax);
			sum += input->matrix[i];
		}
		for (int i = 0; i < total_data_size; i++) {
			input->matrix[i] /= sum;
		}

		return input;
	}

	SoftmaxActivationLayer::~SoftmaxActivationLayer() {
		// TODO Auto-generated destructor stub
	}

}