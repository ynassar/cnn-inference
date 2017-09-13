#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class SigmoidActivationLayer :
		public Layer
	{
	public:
		CNNInference::Mat<float>* forward(CNNInference::Mat<float>* input);
		SigmoidActivationLayer(void);
		~SigmoidActivationLayer(void);
	};
}
