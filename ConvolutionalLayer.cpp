#include "ConvolutionalLayer.h"
#include "Matrix.h"
#include <iostream>
#include "Matrix.cpp"

namespace CNNInference {

	ConvolutionalLayer::ConvolutionalLayer(ThreeDimensionalArray* filters, float* biases, int filter_size, int output_depth, int input_depth, int input_height, int input_width, int stride, int padding)
	{
		this->raw_filters = filters;
		this->filter_size = filter_size;
		this->input_depth = input_depth;
		this->output_depth = output_depth;
		this->input_height = input_height;
		this->input_width = input_width;
		this->stride = stride;
		this->padding = padding;
		this->im2col_kernal(filters, input_depth, output_depth, filter_size);
		this->output_height = (this->input_height - this->filter_size) / this->stride + 1;
		this->output_width = (this->input_width - this->filter_size) / this->stride + 1;
		this->img_transformed = new Matrix<float>(filter_size * filter_size * input_depth, output_height * output_width, 8);
		this->output = new Matrix<float>(this->kernels->height, this->img_transformed->width, 8);
		for (int i = 0; i < this->kernels->height; i++) {
			for (int j = 0; j < this->img_transformed->width; j++) {
				(*this->output)[i][j] = 0;
			}
		}
		this->biases = new Matrix<float>(*this->output);
		for (int i = 0; i < this->biases->height; i++) {
			for (int j = 0; j < this->biases->width; j++) {
				(*this->biases)[i][j] = biases[i];
			}
		}
	}

	Matrix<float>* ConvolutionalLayer::forward(Matrix<float>* input) {
		im2col(*input);
		this->kernels->mult3(*this->img_transformed, this->output);
		this->output->element_wise_add_AVX(*this->biases);
		return this->output;
	}

	void ConvolutionalLayer::im2col_kernal(ThreeDimensionalArray* filters,
		int in_c, int out_c, int ker_size)
	{
		this->kernels = new Matrix<float>(out_c, ker_size*ker_size*in_c, 8);
		int count = 0;
		for (int i = 0; i < out_c; i++) {
			count = 0;
			for (int j = 0; j < in_c; j++)
				for (int k = 0; k < ker_size; k++)
					for (int l = 0; l < ker_size; l++)
					{
						this->kernels->matrix[i*kernels->phy_width + count] = filters[i].element_at(j, k, l);
						count++;
					}
		}
	}

	void ConvolutionalLayer::im2col(Matrix<float>& images) {
		uint16_t h1 = this->output_height;
		uint16_t w1 = this->output_width;
		uint16_t len = this->input_depth*this->filter_size*this->filter_size*h1*w1;
		uint16_t p, q, d0, i, j;
		uint16_t x = h1*w1;
		uint16_t f = this->filter_size;
		uint16_t phyw = this->img_transformed->phy_width;
		uint16_t inw = this->input_width;
		for (uint16_t k = 0; k < len; k+=1)
		{
			p = k / (x);
			q = k % (x);
			d0 = (p / f) / f;
			i = (q / w1) + (p / f) % f;
			j = (q%w1) + (p%f);
			this->img_transformed->matrix[p*phyw + q] = images[d0][i*inw + j];
		}
	}

	ConvolutionalLayer::~ConvolutionalLayer(void)
	{
	}
}