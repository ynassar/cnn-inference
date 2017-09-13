#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class ReluActivationLayer : public Layer
	{
	public:
		CNNInference::Mat<float>* forward(CNNInference::Mat<float>* input);
		ReluActivationLayer(void);
		~ReluActivationLayer(void);
	};
}

