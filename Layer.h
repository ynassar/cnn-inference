#pragma once

#include "ThreeDimensionalArray.h"
#include "Mat.h"

namespace CNNInference{

	class Layer
	{
	public:
		Layer(void);
		virtual Utils::Mat<float>* forward(Utils::Mat<float>* input) = 0;
		~Layer(void);
	};

}
