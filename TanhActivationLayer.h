#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class TanhActivationLayer :
		public Layer
	{
	public:
		TanhActivationLayer(void);
		CNNInference::Mat<float>* forward(CNNInference::Mat<float>* input);
		~TanhActivationLayer(void);
	};

}
