/*
 * PoolingLayer.h
 *
 *  Created on: Sep 10, 2017
 *      Author: yousefnassar
 */

#ifndef POOLINGLAYER_H_
#define POOLINGLAYER_H_

#include "Layer.h"

class PoolingLayer: public Layer {
public:
	int kernel_size, stride;
	PoolingLayer();
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	virtual ~PoolingLayer();
};

#endif /* POOLINGLAYER_H_ */
