#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class TanhActivationLayer :
		public Layer
	{
	public:
		TanhActivationLayer(void);
		CNNInference::Matrix<float>* forward(CNNInference::Matrix<float>* input);
		~TanhActivationLayer(void);
	};

}
