#include "ThreeDimensionalArray.h"
#include <cstring>

namespace CNNInference {
	ThreeDimensionalArray::ThreeDimensionalArray(void)
	{
	}

	ThreeDimensionalArray::ThreeDimensionalArray(int depth, int height, int width) {
		this->depth = depth;
		this->height = height;
		this->width = width;
		this->data = new float[depth*height*width];
		memset(this->data, 0, sizeof(float) * depth * height * width);
	}

	float& ThreeDimensionalArray::element_at(int i, int j, int k) {
		return this->data[i * this->height * this->width + j * this->width + k];
	}


	ThreeDimensionalArray::~ThreeDimensionalArray(void)
	{
	}
}