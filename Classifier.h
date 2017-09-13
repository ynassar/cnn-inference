#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include <vector>
#include <string>
#include "Mat.h"

namespace CNNInference{
	class Classifier
	{
		std::vector<Layer*> layers;
	public:
		Classifier(const std::string& descriptor_file);
		~Classifier(void);

		CNNInference::Mat<float>* predict(CNNInference::Mat<float>* input);
		CNNInference::Mat<float>* MatrixFromFile(const std::string& txtfile);
	};

}
