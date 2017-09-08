#pragma once

#include "Matrix.h"
#include "Vector.h"

class NaiveMatrixMultiplier
{
public:
	static Vector* vector_matrix_multiply(Vector* vector, Matrix* matrix);
	NaiveMatrixMultiplier(void);
	~NaiveMatrixMultiplier(void);
};

