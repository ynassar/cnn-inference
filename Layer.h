#pragma once

#include "ThreeDimensionalArray.h"
#include "Mat.h"

namespace CNNInference{

	class Layer
	{
	public:
		Layer(void);
		virtual CNNInference::Mat<float>* forward(CNNInference::Mat<float>* input) = 0;
		~Layer(void);
	};

}
