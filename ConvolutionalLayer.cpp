#include "ConvolutionalLayer.h"
#include "NaiveConvolutor.h"


ConvolutionalLayer::ConvolutionalLayer(void)
{
}

ThreeDimensionalArray* ConvolutionalLayer::forward(ThreeDimensionalArray* input){
	int output_height = (input->height - this->filter_size + 2 * this->padding) / this->stride + 1;
	int output_width = (input->width - this->filter_size + 2 * this->padding) / this->stride + 1;
	ThreeDimensionalArray* out_matrices = new ThreeDimensionalArray(output_depth, output_height, output_width);
	for (int i = 0; i < this->output_depth; ++i){
		ThreeDimensionalArray* filters_for_output = this->filters + i;
		Matrix* out_matrix = out_matrices->matrix_at(i);
		for (int j = 0; j < input_depth; ++j){
			Matrix* filter = filters_for_output->matrix_at(j);
			Matrix* corresponding_channel = input->matrix_at(j);
			corresponding_channel->pad_zeros(padding, padding);
			NaiveConvolutor::convolute_and_add(corresponding_channel, filter, out_matrix);
		}
	}

	return out_matrices;
}

ConvolutionalLayer::~ConvolutionalLayer(void)
{
}
