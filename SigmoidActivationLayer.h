#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class SigmoidActivationLayer :
		public Layer
	{
	public:
		CNNInference::Matrix<float>* forward(CNNInference::Matrix<float>* input);
		SigmoidActivationLayer(void);
		~SigmoidActivationLayer(void);
	};
}
