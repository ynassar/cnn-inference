#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class SigmoidActivationLayer :
		public Layer
	{
	public:
		Utils::Mat<float>* forward(Utils::Mat<float>* input);
		SigmoidActivationLayer(void);
		~SigmoidActivationLayer(void);
	};
}
