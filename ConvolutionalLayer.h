#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include "Matrix.h"

namespace CNNInference {
	class ConvolutionalLayer : public Layer
	{
	public:
		CNNInference::Matrix<float>* kernels;
		CNNInference::Matrix<float>* output;
		CNNInference::Matrix<float>* biases;
		CNNInference::Matrix<float>* img_transformed;
		ThreeDimensionalArray* raw_filters;
		int input_depth;
		int output_depth;
		int filter_size;
		int input_height;
		int input_width;
		int output_height;
		int output_width;
		int stride;
		int padding;
		CNNInference::Matrix<float>* forward(CNNInference::Matrix<float>* input);
		ConvolutionalLayer(ThreeDimensionalArray* filters, float* biases, int filter_size, int output_depth, int input_depth, int input_height, int input_width, int stride, int padding);
		~ConvolutionalLayer(void);
	private:
		void im2col_kernal(ThreeDimensionalArray* kernal, int in_c,int out_c, int ker_size);
		void im2col(CNNInference::Matrix<float>& images);
	};

}
