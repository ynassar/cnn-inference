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
		Matrix<float>* mean_image;
		int first_layer_input_depth, first_layer_input_height, first_layer_input_width;
	public:
		Classifier(const std::string& descriptor_file);
		Classifier(const std::string& descriptor_file, const std::string& mean_img_file);
		~Classifier(void);
		Matrix<float>* prepare_input(float* input);
		Matrix<float>* prepare_input(unsigned char* input);
		void LoadMeanImage(const std::string& mean_img_file);
		void LoadDescriptor(const std::string& descriptor_file);
		Matrix<float>* predict(CNNInference::Matrix<float>* input);
	};

}
