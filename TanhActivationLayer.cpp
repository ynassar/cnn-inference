#include "TanhActivationLayer.h"
#include <cmath>
using namespace CNNInference;

TanhActivationLayer::TanhActivationLayer(void)
{
}

CNNInference::Mat<float>* TanhActivationLayer::forward(CNNInference::Mat<float>* input){

	int limit = input->phy_height * input->phy_width;
	for (int i = 0; i < limit; i ++){
		input->matrix[i] = tanh(input->matrix[i]);
	}

	return input;
}

TanhActivationLayer::~TanhActivationLayer(void)
{
}
