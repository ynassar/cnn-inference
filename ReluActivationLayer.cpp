#include "ReluActivationLayer.h"
#include <algorithm>
#include "Mat.h"
#include "Mat.cpp"
#include <iostream>
using namespace std;
using namespace CNNInference;


CNNInference::Matrix<float>* ReluActivationLayer::forward(CNNInference::Matrix<float>* input){
	int limit = input->phy_height * input->phy_width;
	for (int i = 0; i < limit; i ++){
		input->matrix[i] = max(input->matrix[i], 0.f);
	}
	return input;
}

ReluActivationLayer::ReluActivationLayer(void)
{
}


ReluActivationLayer::~ReluActivationLayer(void)
{
}
