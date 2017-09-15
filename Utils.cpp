#include "Utils.h"
#include <fstream>
#include <iostream>
#include <string>

namespace CNNInference {
	Matrix<float>* matrix_from_file(const std::string& txtfile) {
		std::ifstream infile(txtfile);
		int channels_in, height, width;
		infile >> channels_in >> height >> width;
		Matrix<float>* input_mat = new Matrix<float>(channels_in, height * width, 1);
		int temp;
		float* data = new float[channels_in * height * width];
		for (int i = 0; i < channels_in * height * width; i++) {
			infile >> data[i];
		}

		for (int i = 0; i < channels_in; i++) {
			for (int j = 0; j < height; j++) {
				for (int k = 0; k < width; k++) {
					input_mat->operator [](i)[j*width + k] = data[i * width * height + j * width + k];
				}
			}
		}
		return input_mat;
	}

	// Image : in_channels x in_height x in_width
	// Out Transformed: in_channels x num_tiles_h * 4 x num_tiles_w * 4
	void tile_and_transform_2_3(float* image, float* out_transformed, int in_channels, int in_height, int in_width) {
		int num_tiles_h = 1 + (in_height - 1) / 2;
		int num_tiles_w = 1 + (in_width - 1) / 2;
		int out_channels = in_channels;
		int out_h = num_tiles_h * 4;
		int out_w = num_tiles_w * 4;

		float tmp[16];
		float s[16];

		float* current_image;
		float* channel_output;
		float* current_output;
		for (int c = 0; c < in_channels; c++) {
			current_image = image + c * in_height * in_width;
			channel_output = out_transformed + c * out_h * out_w;
			for (int i = 0; i < num_tiles_h; i += 1) {
				for (int j = 0; j < num_tiles_w; j += 1) {
					current_output = channel_output + 4 * 4 * ((i * num_tiles_w) + j);
					tmp[0] = current_image[2*i * in_width + 2*j]; tmp[1] = current_image[2*i * in_width + 2*j + 1]; tmp[2] = current_image[2*i * in_width + 2*j + 2]; tmp[3] = current_image[2*i * in_width + 3];
					tmp[4] = current_image[(2*i + 1) * in_width + 2*j]; tmp[5] = current_image[(2*i + 1) * in_width + 2*j + 1]; tmp[6] = current_image[(2*i + 1) * in_width + 2*j + 2]; tmp[7] = current_image[(2*i + 1) * in_width + 2*j + 3];
					tmp[8] = current_image[(2*i + 2) * in_width + 2*j]; tmp[9] = current_image[(2*i + 2) * in_width + 2*j + 1]; tmp[10] = current_image[(2*i + 2) * in_width + 2*j + 2]; tmp[11] = current_image[(2*i + 2) * in_width + 2*j + 3];
					tmp[12] = current_image[(2*i + 3) * in_width + 2*j]; tmp[13] = current_image[(2*i + 3) * in_width + 2*j + 1]; tmp[14] = current_image[(2*i + 3) * in_width + 2*j + 2]; tmp[15] = current_image[(2*i + 3) * in_width + 2*j + 3];

					s[0] = (tmp[0] - tmp[8]) - (tmp[2] - tmp[10]);
					s[1] = (tmp[1] - tmp[9]) + (tmp[2] - tmp[10]);
					s[2] = (tmp[2] - tmp[10]) - (tmp[1] - tmp[9]);
					s[3] = (tmp[1] - tmp[9]) - (tmp[3] - tmp[11]);
					s[4] = (tmp[4] + tmp[8]) - (tmp[6] + tmp[10]);
					s[5] = (tmp[5] + tmp[9]) + (tmp[6] + tmp[10]);
					s[6] = (tmp[6] + tmp[10]) - (tmp[5] + tmp[9]);
					s[7] = (tmp[5] + tmp[9]) - (tmp[7] + tmp[11]);
					s[8] = (tmp[8] - tmp[4]) - (tmp[10] - tmp[6]);
					s[9] = (tmp[9] - tmp[5]) + (tmp[10] - tmp[6]);
					s[10] = (tmp[10] - tmp[6]) - (tmp[9] - tmp[5]);
					s[11] = (tmp[9] - tmp[5]) - (tmp[11] - tmp[7]);
					s[12] = (tmp[4] - tmp[12]) - (tmp[6] - tmp[14]);
					s[13] = (tmp[5] - tmp[13]) + (tmp[6] - tmp[14]);
					s[14] = (tmp[6] - tmp[14]) - (tmp[5] - tmp[13]);
					s[15] = (tmp[5] - tmp[13]) - (tmp[7] - tmp[15]);

					current_output[0] = s[0];
					current_output[1] = s[1];
					current_output[2] = s[2];
					current_output[3] = s[3]; 
					current_output[4] = s[4];
					current_output[5] = s[5];
					current_output[6] = s[6];
					current_output[7] = s[7];
					current_output[8] = s[8];
					current_output[9] = s[9];
					current_output[10] = s[10];
					current_output[11] = s[11];
					current_output[12] = s[12];
					current_output[13] = s[13];
					current_output[14] = s[14];
					current_output[15] = s[15];

				}
			}
		}
	}

	// In: kernels out_channels x in_channels x 3 x 3
	// Out: output: out_channels x in_channels x 4 x 4
	void transform_kernels_2_3(float* kernels, float* output, int out_channels, int in_channels) {
		for (int oc = 0; oc < out_channels; oc += 1) {
			float* kernels_for_oc = kernels + oc * in_channels * 3 * 3;
			float* outputs_for_oc = output + oc * in_channels * 4 * 4;
			for (int ic = 0; ic < in_channels; ic += 1) {
				float* current_kernel = kernels_for_oc + ic * 3 * 3;
				float* current_output = outputs_for_oc + ic * 4 * 4;
				current_output[0] = current_kernel[0];
				current_output[1] = (current_kernel[0] + current_kernel[2] + current_kernel[1])*0.5f;
				current_output[2] = (current_kernel[0] + current_kernel[2] - current_kernel[1])*0.5f;
				current_output[3] = current_kernel[2];
				current_output[4] = (current_kernel[0] + current_kernel[6] + current_kernel[3])*0.5f;
				current_output[5] = ((current_kernel[0] + current_kernel[6] + current_kernel[3]) + (current_kernel[2] + current_kernel[8] + current_kernel[5]) + (current_kernel[1] + current_kernel[7] + current_kernel[4]))*0.25f;
				current_output[6] = ((current_kernel[0] + current_kernel[6] + current_kernel[3]) + (current_kernel[2] + current_kernel[8] + current_kernel[5]) - (current_kernel[1] + current_kernel[7] + current_kernel[4]))*0.25f;
				current_output[7] = (current_kernel[2] + current_kernel[8] + current_kernel[5])*0.5f;
				current_output[8] = (current_kernel[0] + current_kernel[6] - current_kernel[3])*0.5f;
				current_output[9] = ((current_kernel[0] + current_kernel[6] - current_kernel[3]) + (current_kernel[2] + current_kernel[8] - current_kernel[5]) + (current_kernel[1] + current_kernel[7] - current_kernel[4]))*0.25f;
				current_output[10] = ((current_kernel[0] + current_kernel[6] - current_kernel[3]) + (current_kernel[2] + current_kernel[8] - current_kernel[5]) - (current_kernel[1] + current_kernel[7] - current_kernel[4]))*0.25f;
				current_output[11] = (current_kernel[2] + current_kernel[8] - current_kernel[5])*0.5f;
				current_output[12] = current_kernel[6];
				current_output[13] = (current_kernel[6] + current_kernel[8] + current_kernel[7])*0.5f;
				current_output[14] = (current_kernel[6] + current_kernel[8] - current_kernel[7])*0.5f;
				current_output[15] = current_kernel[8];
			}
		}
	}

	// In: transformed_image : in_channels x num_tiles_h x num_tiles_w x 16
	// In: transformed_kernels: out_channels x in_channels x 16
	// Out: output: out_channels x num_tiles_h x num_tiles_w x 16
	void elementwise_mult_and_add(float* transformed_image, float* transformed_kernels, float* output, int in_channels, int out_channels, int in_height, int in_width) {
		int num_tiles_h = 1 + (in_height - 1) / 2;
		int num_tiles_w = 1 + (in_width - 1) / 2;
		for (int out_ch = 0; out_ch < out_channels; out_ch += 1) {
			float* channel_output = output + out_ch * num_tiles_h * num_tiles_w * 16;
			float* channel_kernels = transformed_kernels + out_ch * in_channels * 16;
			for (int in_ch = 0; in_ch < in_channels; in_ch += 1) {
				float* current_kernel = channel_kernels + in_ch * 16;
				float* current_img_channel = transformed_image + num_tiles_h * num_tiles_w * 16 * in_ch;
				for (int tile_h = 0; tile_h < num_tiles_h; tile_h += 1) {
					for (int tile_w = 0; tile_w < num_tiles_w; tile_w += 1) {
						float* current_tile = current_img_channel + (tile_h * num_tiles_w + tile_w) * 16;
						float* tile_output = channel_output + (tile_h * num_tiles_w + tile_w) * 16;
						for (int el = 0; el < 16; el+=1) {
							tile_output[el] += current_kernel[el] * current_tile[el];
						}
					}
				}
			}
		}
	}

	// In: multiplied_input : out_channels x num_tiles_h x num_tiles_w x 16
	void inverse_transform(float* multiplied_input, float* output, int out_channels, int in_height, int in_width) {
		int num_tiles_h = 1 + (in_height - 1) / 2;
		int num_tiles_w = 1 + (in_width - 1) / 2;
		float temp[8];
		for (int out_ch = 0; out_ch < out_channels; out_ch += 1) {
			float* channel_input = multiplied_input + out_ch * num_tiles_h * num_tiles_w * 16;
			float* channel_output = output + out_ch * num_tiles_h * num_tiles_w * 4;
			for (int tile_h = 0; tile_h < num_tiles_h; tile_h += 1) {
				for (int tile_w = 0; tile_w < num_tiles_w; tile_w += 1) {
					float* tile_input = channel_input + (tile_h * num_tiles_w + tile_w) * 16;
					temp[0] = tile_input[0] + tile_input[1] + tile_input[2];
					temp[1] = tile_input[1] - tile_input[2] - tile_input[3];
					temp[2] = tile_input[4] + tile_input[5] + tile_input[6];
					temp[3] = tile_input[5] - tile_input[6] - tile_input[7];
					temp[4] = tile_input[8] + tile_input[9] + tile_input[10];
					temp[5] = tile_input[9] - tile_input[10] - tile_input[11];
					temp[6] = tile_input[12] + tile_input[13] + tile_input[14];
					temp[7] = tile_input[13] - tile_input[14] - tile_input[15];
					channel_output[tile_h*2*2*num_tiles_w + 2*tile_w] = temp[0] + temp[2] + temp[4];
					channel_output[tile_h*2*2 * num_tiles_w + 2 * tile_w + 1] = temp[1] + temp[3] + temp[5];
					channel_output[(tile_h * 2 + 1) * 2 * num_tiles_w + 2 * tile_w] = temp[2] - temp[4] - temp[6];
					channel_output[(tile_h * 2 +1)* 2 * num_tiles_w + 2 * tile_w + 1] = temp[3] - temp[5] - temp[7];
				}
			}
		}
	}


}