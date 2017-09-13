#pragma once

#include "Layer.h"
#include "Mat.h"

namespace CNNInference{
	class ReluActivationLayer : public Layer
	{
	public:
		CNNInference::Matrix<float>* forward(CNNInference::Matrix<float>* input);
		ReluActivationLayer(void);
		~ReluActivationLayer(void);
	};
}

