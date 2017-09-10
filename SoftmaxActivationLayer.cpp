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
using namespace std;

SoftmaxActivationLayer::SoftmaxActivationLayer() {
	// TODO Auto-generated constructor stub

}

ThreeDimensionalArray* SoftmaxActivationLayer::forward(ThreeDimensionalArray* input){
	float ymax = -std::numeric_limits<float>::max();
	int total_data_size = input->depth * input->height * input->width;
	for (int i = 0; i < total_data_size; i ++){
		ymax = max(ymax, input->data[i]);
	}
	float sum = 0;
	for (int i = 0; i < total_data_size; i ++){
		input->data[i] = exp(input->data[i] - ymax);
		sum += input->data[i];
	}
	for (int i = 0; i < total_data_size; i ++){
		input->data[i] /= sum;
	}
	return input;
}

SoftmaxActivationLayer::~SoftmaxActivationLayer() {
	// TODO Auto-generated destructor stub
}

