#include "Classifier.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "TanhActivationLayer.h"
#include "SigmoidActivationLayer.h"

Classifier::Classifier(void)
{
}

Vector* Classifier::predict(ThreeDimensionalArray* input){
	ThreeDimensionalArray* last_layer_output = input;
	for (int i = 0; i < this->layers.size(); ++i){
		last_layer_output = this->layers[i]->forward(last_layer_output);
	}

	return last_layer_output->to_vector();
}

Classifier::~Classifier(void)
{
}
