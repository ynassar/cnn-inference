#include "ConvolutionalLayer.h"
#include "NaiveConvolutor.h"


ConvolutionalLayer::ConvolutionalLayer(void)
{
}

ThreeDimensionalArray* ConvolutionalLayer::forward(ThreeDimensionalArray* input){
	Matrix* out_matrices = new Matrix[output_depth];
	for (int i = 0; i < this->output_depth; i ++){
		ThreeDimensionalArray* filters_for_output = this->filters + i;
		int output_height = (input->matrix_at(0)->height - this->filter_size + 2 * this->padding) / this->stride + 1;
		int output_width = (input->matrix_at(0)->width - this->filter_size + 2 * this->padding) / this->stride + 1;
		out_matrices[i] = Matrix(output_height, output_width);
		for (int j = 0; j < input_depth; j ++){
			Matrix* filter = filters_for_output->matrix_at(j);
			Matrix* corresponding_channel = input->matrix_at(j);
			corresponding_channel->pad_zeros(padding, padding);
			NaiveConvolutor::convolute_and_add(corresponding_channel, filter, out_matrices + i);
		}
	}

	return new ThreeDimensionalArray(out_matrices, output_depth);
}

ConvolutionalLayer::~ConvolutionalLayer(void)
{
}
