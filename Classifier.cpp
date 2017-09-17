#include "Classifier.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "TanhActivationLayer.h"
#include "SigmoidActivationLayer.h"
#include "SoftmaxActivationLayer.h"
#include "ReluActivationLayer.h"
#include "PoolingLayer.h"
#include <vector>
#include <fstream>
#include <iostream>
#include "Matrix.h"
#include "Matrix.cpp"

namespace CNNInference {

	Classifier::Classifier(const std::string & descriptor_file)
	{
		this->mean_image = NULL;
		this->LoadDescriptor(descriptor_file);
	}

	Classifier::Classifier(const std::string& descriptor_file, const std::string& mean_img_file)
	{
		this->LoadDescriptor(descriptor_file);
		this->LoadMeanImage(mean_img_file);
	}

	Matrix<float>* Classifier::predict(Matrix<float>* input) {
		Matrix<float>* last_layer_output = input;
		for (int i = 0; i < (int)this->layers.size(); ++i) {
			last_layer_output = this->layers[i]->forward(last_layer_output);
		}
		return last_layer_output;
	}

	Classifier::~Classifier(void)
	{
		this->layers.clear();
	}

	Matrix<float>* Classifier::prepare_input(float* input)
	{
		Matrix<float>* input_matrix = new Matrix<float>(this->first_layer_input_depth, this->first_layer_input_height* this->first_layer_input_width, 8);
		if (this->mean_image != NULL) {
			for (int i = 0; i < this->first_layer_input_depth; i+=1) {
				for (int j = 0; j < this->first_layer_input_height; j+=1) {
					for (int k = 0; k < this->first_layer_input_width; k+=1) {
						(*input_matrix)[i][j * this->first_layer_input_width + k] = input[i * this->first_layer_input_height * this->first_layer_input_width + j * this->first_layer_input_width + k] - (*this->mean_image)[i][j * this->first_layer_input_width + k];
					}
				}
			}
		}
		else {
			for (int i = 0; i < this->first_layer_input_depth; i+=1) {
				for (int j = 0; j < this->first_layer_input_height; j+=1) {
					for (int k = 0; k < this->first_layer_input_width; k+=1) {
						(*input_matrix)[i][j * this->first_layer_input_width + k] = input[i * this->first_layer_input_height * this->first_layer_input_width + j * this->first_layer_input_width + k];
					}
				}
			}
		}
		return input_matrix;
	}

	Matrix<float>* Classifier::prepare_input(unsigned char* input)
	{
		Matrix<float>* input_matrix = new Matrix<float>(this->first_layer_input_depth, this->first_layer_input_height * this->first_layer_input_width, 8);
		if (this->mean_image != NULL) {
			for (int i = 0; i < this->first_layer_input_depth; i+=1) {
				for (int j = 0; j < this->first_layer_input_height; j+=1) {
					for (int k = 0; k < this->first_layer_input_width; k+=1) {
						(*input_matrix)[i][j * this->first_layer_input_width + k] = (float)input[i * this->first_layer_input_height * this->first_layer_input_width + j * this->first_layer_input_width + k] - (*this->mean_image)[i][j * this->first_layer_input_width + k];
					}
				}
			}
		}
		else {
			for (int i = 0; i < this->first_layer_input_depth; i+=1) {
				for (int j = 0; j < this->first_layer_input_height; j+=1) {
					for (int k = 0; k < this->first_layer_input_width; k+=1) {
						(*input_matrix)[i][j * this->first_layer_input_width + k] = (float)input[i * this->first_layer_input_height * this->first_layer_input_width + j * this->first_layer_input_width + k];
					}
				}
			}
		}
		return input_matrix;
	}

	void Classifier::LoadMeanImage(const std::string & mean_img_file)
	{
		std::ifstream infile(mean_img_file.c_str());
		int depth, height, width;
		infile >> depth >> height >> width;
		this->mean_image = new Matrix<float>(depth, height * width, 8);
		for (int i = 0; i < depth; i++) {
			for (int j = 0; j < height; j++) {
				for (int k = 0; k < width; k++) {
					float element; infile >> element;
					(*this->mean_image)[i][j * width + k] = element;
				}
			}
		}
	}
	void Classifier::LoadDescriptor(const std::string & descriptor_file)
	{
		std::ifstream infile(descriptor_file.c_str());
		int input_n, input_depth, input_height, input_width;

		infile >> input_n >> input_depth >> input_height >> input_width;
		this->first_layer_input_depth = input_depth;
		this->first_layer_input_height = input_height;
		this->first_layer_input_width = input_width;
		while (!infile.eof()) {
			std::string layer_type;
			infile >> layer_type;
			if (layer_type == "Convolution") {
				std::string layer_name;
				infile >> layer_name;
				std::cout << "Building layer " << layer_name << std::endl;
				int stride;
				int padding;
				infile >> stride >> padding;
				int in_channels, out_channels, kernel_height, kernel_width;
				infile >> out_channels >> in_channels >> kernel_height >> kernel_width;
				float* layer_params = new float[in_channels * out_channels * kernel_height * kernel_width];

				for (int i = 0; i < in_channels * out_channels * kernel_height * kernel_width; i++) {
					infile >> layer_params[i];
				}

				int bias_size; infile >> bias_size;

				float* bias = new float[bias_size];
				for (int i = 0; i < bias_size; i++) {
					infile >> bias[i];
				}

				ThreeDimensionalArray* filters = new ThreeDimensionalArray[out_channels];
				for (int i = 0; i < out_channels; i++) {
					filters[i].height = kernel_height;
					filters[i].depth = in_channels;
					filters[i].width = kernel_width;
					filters[i].data = layer_params;
					layer_params += in_channels * kernel_height * kernel_width;
				}

				if (in_channels != input_depth) {
					std::cout << "Error reading model, in_channels for layer " << layer_name << " = " << in_channels << " != out channels for previous layer = " << input_depth << std::endl;
				}

				ConvolutionalLayer* new_layer = new ConvolutionalLayer(filters, bias, kernel_height, out_channels, in_channels, input_height, input_width, stride, padding);
				this->layers.push_back(new_layer);
				input_height = new_layer->output_height;
				input_width = new_layer->output_width;
				input_depth = new_layer->output_depth;
			}
			else if (layer_type == "ReLU") {
				std::string layer_name; infile >> layer_name;
				std::cout << "Building layer " << layer_name << std::endl;
				this->layers.push_back(new ReluActivationLayer());
			}
			else if (layer_type == "InnerProduct") {
				std::string layer_name; infile >> layer_name;
				std::cout << "Building layer " << layer_name << std::endl;

				int input_size, output_size; infile >> input_size >> output_size;
				float* weights = new float[input_size * output_size];
				for (int i = 0; i < input_size * output_size; i++) {
					infile >> weights[i];
				}
				int bias_size; infile >> bias_size;
				float* bias = new float[bias_size];
				for (int i = 0; i < bias_size; i++) {
					infile >> bias[i];
				}

				Matrix<float>* weights_mat = new Matrix<float>(input_size, output_size, 8);
				Matrix<float>* bias_mat = new Matrix<float>(1, bias_size, 8);

				for (int i = 0; i < input_size; i++) {
					for (int j = 0; j < output_size; j++) {
						(*weights_mat)[i][j] = weights[i*output_size + j];
					}
				}

				for (int i = 0; i < bias_size; i++) {
					bias_mat->operator [](0)[i] = bias[i];
				}

				FullyConnectedLayer* new_layer = new FullyConnectedLayer(input_size, output_size, weights_mat, bias_mat);
				this->layers.push_back(new_layer);
			}
			else if (layer_type == "Pooling") {
				std::string layer_name; infile >> layer_name;
				std::cout << "Building layer " << layer_name << std::endl;

				int kernel_size, stride; infile >> kernel_size >> stride;
				PoolingLayer* new_layer = new PoolingLayer(input_depth, input_height, input_width, kernel_size, stride);
				new_layer->kernel_size = kernel_size;
				new_layer->stride = stride;
				this->layers.push_back(new_layer);
				input_height = new_layer->output_height;
				input_width = new_layer->output_width;
			}
			else if (layer_type == "Softmax") {
				std::string layer_name; infile >> layer_name;
				std::cout << "Building layer " << layer_name << std::endl;
				this->layers.push_back(new SoftmaxActivationLayer());
			}
		}
	}
}