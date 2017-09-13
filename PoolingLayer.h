/*
 * PoolingLayer.h
 *
 *  Created on: Sep 10, 2017
 *      Author: yousefnassar
 */

#ifndef POOLINGLAYER_H_
#define POOLINGLAYER_H_

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{

	class PoolingLayer: public Layer {
	public:
		int kernel_size, stride;
		int input_height;
		int input_width;
		int input_depth;
		int output_height;
		int output_width;
		CNNInference::Matrix<float>* output;
		PoolingLayer(int input_depth, int input_height, int input_width, int kernel_size, int stride);
		CNNInference::Matrix<float>* forward(CNNInference::Matrix<float>* input);
		virtual ~PoolingLayer();
	};
}
#endif /* POOLINGLAYER_H_ */
