#pragma once

#include "Vector.h"
#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include <vector>
#include <string>

namespace CNNInference{

	class Classifier
	{
		std::vector<Layer*> layers;
	public:
		Classifier(const std::string& descriptor_file);
		~Classifier(void);

		Vector* predict(ThreeDimensionalArray* input);
	};

}
