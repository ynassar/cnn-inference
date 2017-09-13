#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include <vector>
#include <string>
#include "Matrix.h"

namespace CNNInference{
	class Classifier
	{
		std::vector<Layer*> layers;
	public:
		Classifier(const std::string& descriptor_file);
		~Classifier(void);

		CNNInference::Matrix<float>* predict(CNNInference::Matrix<float>* input);
		CNNInference::Matrix<float>* MatrixFromFile(const std::string& txtfile);
	};

}
