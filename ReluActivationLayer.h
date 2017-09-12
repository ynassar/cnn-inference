#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class ReluActivationLayer : public Layer
	{
	public:
		Utils::Mat<float>* forward(Utils::Mat<float>* input);
		ReluActivationLayer(void);
		~ReluActivationLayer(void);
	};
}

