#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class TanhActivationLayer :
		public Layer
	{
	public:
		TanhActivationLayer(void);
		Utils::Mat<float>* forward(Utils::Mat<float>* input);
		~TanhActivationLayer(void);
	};

}
