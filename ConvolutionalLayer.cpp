#include "ConvolutionalLayer.h"
#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include<chrono>
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
		std::cout << "Entering im2col kernal filter_size = " << filter_size << std::endl;
		this->im2row_kernal(filters, input_depth, output_depth, filter_size);
		this->output_height = (this->input_height - this->filter_size + padding + padding) / this->stride + 1;
		this->output_width = (this->input_width - this->filter_size + padding + padding) / this->stride + 1;
		this->img_transformed = new Matrix<float>(output_height * output_width, filter_size * filter_size * input_depth, 8);
		this->output = new Matrix<float>(this->img_transformed->height, this->kernels->width, 8);
		this->output_transposed = new Matrix<float>(this->kernels->width, this->img_transformed->height, 8);
		int siz = ROUND_UP(filter_size*filter_size, 8);
		this->filter = (float*)_mm_malloc(sizeof(float)*siz, 64);
		memset(filter, 0, sizeof(float)*siz);
		output->fill_zeros();
		img_transformed->fill_zeros();
		this->biases = new Matrix<float>(this->output_transposed->height, this->output_transposed->width, 8);
		for (int i = 0; i < this->biases->height; i++) {
			for (int j = 0; j < this->biases->width; j++) {
				(*this->biases)[i][j] = biases[i];
			}
		}
	}

	Matrix<float>* ConvolutionalLayer::forward(Matrix<float>* input) {
		im2row(*input);
		img_transformed->mult3(*kernels, output);
		output->transpose(output_transposed);
		this->output_transposed->element_wise_add_AVX(*this->biases);
		return this->output_transposed;
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

		if (output_width > 8) {
			__m256 ymm0;
			int h1 = this->output_height;
			int w1 = this->output_width;
			int len = this->input_depth*this->filter_size*this->filter_size;
			int d0, i, j;
			for (int p = 0; p < len; p++) {
				for (int q = 0; q < h1*w1; q++) {
					d0 = (p / this->filter_size) / this->filter_size;
					i = (q / w1)*stride + ((p / this->filter_size)) % this->filter_size;
					j = (q % w1)*stride + (p % this->filter_size);
					if (~(i < padding || i >= input_height + padding || j < padding || j >= input_width + padding))
						filter[q] = images[d0][(i - padding) * this->input_width + (j - padding)];
					else
						filter[q] = 0.f;
				}
				for (int t = 0; t < img_transformed->phy_width; t += 8) {
					ymm0 = _mm256_load_ps(filter);
					_mm256_store_ps(&(*img_transformed)[p][t], ymm0);
				}

			}
		}
		else {
			int h1 = this->output_height;
			int w1 = this->output_width;
			int len = this->input_depth*this->filter_size*this->filter_size*h1*w1;
			int p, q, d0, i, j;
			for (int k = 0; k < len; k++) {
				p = k / (h1*w1);
				q = k % (h1*w1);
				d0 = (p / this->filter_size) / this->filter_size;
				i = (q / w1)*stride + ((p / this->filter_size)) % this->filter_size;
				j = (q % w1)*stride + (p % this->filter_size);
				if (~(i < padding || i >= input_height + padding || j < padding || j >= input_width + padding))
					this->img_transformed->matrix[p*this->img_transformed->phy_width + q] = images[d0][(i - padding)*this->input_width + (j - padding)];
				else
					this->img_transformed->matrix[p*this->img_transformed->phy_width + q] = 0.f;
			}
		}

	}
	void ConvolutionalLayer::im2row(Matrix<float>& images) {
		__m256 ymm0;
		uint16_t tempi, tempj;
		uint16_t f = this->filter_size;
		uint16_t siz = ROUND_UP(f*f, 8);
		uint16_t pad = this->padding;
		uint16_t in_h = this->input_height;
		uint16_t in_w = this->input_width;
		uint16_t out_w = this->output_width;
		uint16_t f_f = f * f;
		uint16_t c_f_f;
		uint16_t k_f;
		uint16_t tempi_in_w;
		uint16_t in_h_pad = in_h + pad;
		uint16_t in_w_pad = in_w + pad;
		uint16_t i_stride;
		uint16_t j_stride;
		uint16_t i_out_w;
		for (uint16_t c = 0; c < input_depth; ++c) {
			c_f_f = c * f_f;
			for (uint16_t i = 0; i < output_height; ++i) {
				i_stride = i * stride;
				i_out_w = i * out_w;
				for (uint16_t j = 0; j < output_width; ++j) {
					j_stride = j * stride;
					for (uint16_t k = 0; k < filter_size; ++k) {
						tempi = (i_stride + k);
						k_f = k * f;
						tempi_in_w = tempi * in_w;
						for (uint16_t l = 0; l < filter_size; ++l) {
							tempj = j_stride + l;
							if (~(tempi < pad || tempi >= in_h_pad ||
								tempj < pad || tempj >= in_w_pad))
								filter[k_f + l] = images[c][tempi_in_w + tempj];
							else
								filter[k_f + l] = 0.00000001f;

						}
					}
					for (uint16_t t = 0; t < siz; t += 8) {
						ymm0 = _mm256_loadu_ps(filter + t);
						_mm256_storeu_ps(&(*img_transformed)[i_out_w + j][(c_f_f) + t], ymm0);
					}
				}
			}
		}
	}
	void ConvolutionalLayer::im2row_kernal(ThreeDimensionalArray* filters,
		int in_c, int out_c, int ker_size)
	{
		this->kernels = new Matrix<float>(ker_size*ker_size*in_c, out_c, 8);
		int count = 0;
		for (int i = 0; i < out_c; i++) {
			count = 0;
			for (int j = 0; j < in_c; j++)
				for (int k = 0; k < ker_size; k++)
					for (int l = 0; l < ker_size; l++)
					{
						this->kernels->matrix[count*kernels->phy_width + i] = filters[i].element_at(j, k, l);
						count++;
					}
		}
	}

	ConvolutionalLayer::~ConvolutionalLayer(void)
	{
		delete this->biases;
		delete this->filter;
		delete this->img_transformed;
		delete this->output;
		delete this->output_transposed;
		delete this->raw_filters;
	}
}