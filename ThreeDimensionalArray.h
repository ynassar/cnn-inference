#pragma once

#include "Vector.h"
#include "Matrix.h"

class ThreeDimensionalArray
{
public:
	float* data;
	int depth, height, width;

	ThreeDimensionalArray(void);
	ThreeDimensionalArray(int depth, int height, int width);
	ThreeDimensionalArray(Matrix*);
	ThreeDimensionalArray(Vector*);
	Vector* to_vector();
	Matrix* to_matrix();
	float& element_at(int i, int j, int k);
	Matrix* matrix_at(int i);
	~ThreeDimensionalArray(void);
};
