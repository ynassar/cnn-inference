/*
 * SoftmaxActivationLayer.h
 *
 *  Created on: Sep 10, 2017
 *      Author: yousefnassar
 */

#ifndef SOFTMAXACTIVATIONLAYER_H_
#define SOFTMAXACTIVATIONLAYER_H_

#include "Layer.h"

class SoftmaxActivationLayer: public Layer {
public:
	SoftmaxActivationLayer();
	ThreeDimensionalArray* forward(ThreeDimensionalArray* input);
	virtual ~SoftmaxActivationLayer();
};

#endif /* SOFTMAXACTIVATIONLAYER_H_ */
