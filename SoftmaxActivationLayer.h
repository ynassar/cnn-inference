/*
 * SoftmaxActivationLayer.h
 *
 *  Created on: Sep 10, 2017
 *      Author: yousefnassar
 */

#ifndef SOFTMAXACTIVATIONLAYER_H_
#define SOFTMAXACTIVATIONLAYER_H_

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class SoftmaxActivationLayer: public Layer {
	public:
		SoftmaxActivationLayer();
		Utils::Mat<float>* forward(Utils::Mat<float>* input);
		virtual ~SoftmaxActivationLayer();
	};
}
#endif /* SOFTMAXACTIVATIONLAYER_H_ */
