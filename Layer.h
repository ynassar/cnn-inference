#pragma once

#include "ThreeDimensionalArray.h"
#include "Mat.h"

namespace CNNInference{

	class Layer
	{
	public:
		Layer(void);
		virtual CNNInference::Matrix<float>* forward(CNNInference::Matrix<float>* input) = 0;
		~Layer(void);
	};

}
