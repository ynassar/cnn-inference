#include "TanhActivationLayer.h"
#include <cmath>
using namespace CNNInference;

TanhActivationLayer::TanhActivationLayer(void)
{
}

Utils::Mat<float>* TanhActivationLayer::forward(Utils::Mat<float>* input){

	int limit = input->phy_height * input->phy_width;
	for (int i = 0; i < limit; i ++){
		input->matrix[i] = tanh(input->matrix[i]);
	}

	return input;
}

TanhActivationLayer::~TanhActivationLayer(void)
{
}
