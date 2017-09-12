#pragma once

#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include "Mat.h"

namespace CNNInference {
	class ConvolutionalLayer : public Layer
	{
	public:
		Utils::Mat<float>* kernels;
		Utils::Mat<float>* output;
		Utils::Mat<float>* biases;
		Utils::Mat<float>* img_transformed;
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
		Utils::Mat<float>* forward(Utils::Mat<float>* input);
		ConvolutionalLayer(ThreeDimensionalArray* filters, float* biases, int filter_size, int output_depth, int input_depth, int input_height, int input_width, int stride, int padding);
		~ConvolutionalLayer(void);
	private:
		void im2col_kernal(ThreeDimensionalArray* kernal, int in_c,int out_c, int ker_size);
		void im2col(Utils::Mat<float>& images);
	};

}
