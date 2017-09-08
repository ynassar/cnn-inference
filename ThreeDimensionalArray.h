#pragma once

#include "Vector.h"
#include "Matrix.h"

class ThreeDimensionalArray
{
public:
	Matrix* matrices;
	int depth;

	ThreeDimensionalArray(void);
	ThreeDimensionalArray(int height, int width, int depth);
	ThreeDimensionalArray(Matrix*);
	ThreeDimensionalArray(Matrix*, int depth);
	ThreeDimensionalArray(Vector*);
	Vector* to_vector();
	Matrix* to_matrix();
	float& element_at(int i, int j, int k);
	Matrix* matrix_at(int i);
	~ThreeDimensionalArray(void);
};

