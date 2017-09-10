#include "Classifier.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "TanhActivationLayer.h"
#include "SigmoidActivationLayer.h"
#include "SoftmaxActivationLayer.h"
#include "ReluActivationLayer.h"
#include "PoolingLayer.h"
#include "Vector.h"
#include "Matrix.h"
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;
using namespace CNNInference;

Classifier::Classifier(const string& descriptor_file)
{
	ifstream infile(descriptor_file.c_str());
	while(!infile.eof()){
		string layer_type;
		infile >> layer_type;
		if (layer_type == "Convolution"){
			string layer_name;
			infile >> layer_name;
			int stride;
			infile >> stride;
			int in_channels, out_channels, kernel_height, kernel_width;
			infile >> out_channels >> in_channels >> kernel_height >> kernel_width;
			ConvolutionalLayer* new_layer = new ConvolutionalLayer();
			float* layer_params = new float[in_channels * out_channels * kernel_height * kernel_width];

			for (int i = 0; i < in_channels * out_channels * kernel_height * kernel_width; i ++){
				infile >> layer_params[i];
			}

			int bias_size; infile >> bias_size;

			float* bias = new float[bias_size];
			for (int i = 0; i < bias_size; i ++){
				infile >> bias[i];
			}

			ThreeDimensionalArray* filters = new ThreeDimensionalArray[out_channels];
			for (int i = 0; i < out_channels; i ++){
				filters[i].height = kernel_height;
				filters[i].depth = in_channels;
				filters[i].width = kernel_width;
				filters[i].data = layer_params;
				layer_params += in_channels * kernel_height * kernel_width;
			}

			new_layer->biases = bias;
			new_layer->filters = filters;
			new_layer->filter_size = kernel_height;
			new_layer->input_depth = in_channels;
			new_layer->output_depth = out_channels;
			new_layer->stride = stride;
			new_layer->padding = 0;
			this->layers.push_back(new_layer);
		}
		else if(layer_type == "ReLU"){
			string layer_name; infile >> layer_name;
			this->layers.push_back(new ReluActivationLayer());
		}
		else if(layer_type == "InnerProduct"){
			string layer_name; infile >> layer_name;
			int input_size, output_size; infile >> input_size >> output_size;
			float* weights = new float[input_size * output_size];
			for (int i = 0; i < input_size * output_size; i ++){
				infile >> weights[i];
			}
			int bias_size; infile >> bias_size;
			float* bias = new float[bias_size];
			for (int i = 0; i < bias_size; i ++){
				infile >> bias[i];
			}
			FullyConnectedLayer* new_layer = new FullyConnectedLayer();
			new_layer->weights = new Matrix(weights, input_size, output_size);
			new_layer->biases = new Vector(bias, bias_size);
			this->layers.push_back(new_layer);
		}
		else if (layer_type == "Pooling"){
			string layer_name; infile >> layer_name;
			int kernel_size, stride; infile >> kernel_size >> stride;
			PoolingLayer* new_layer = new PoolingLayer();
			new_layer->kernel_size = kernel_size;
			new_layer->stride = stride;
			this->layers.push_back(new_layer);
		}
		else if (layer_type == "Softmax"){
			string layer_name; infile >> layer_name;
			this->layers.push_back(new SoftmaxActivationLayer());
		}
	}
}

Vector* Classifier::predict(ThreeDimensionalArray* input){
	ThreeDimensionalArray* last_layer_output = input;
	for (int i = 0; i < (int) this->layers.size(); ++i){
		last_layer_output = this->layers[i]->forward(last_layer_output);
	}
	return last_layer_output->to_vector();
}

Classifier::~Classifier(void)
{
}
