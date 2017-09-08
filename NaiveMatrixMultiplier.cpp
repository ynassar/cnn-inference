#include "NaiveMatrixMultiplier.h"


NaiveMatrixMultiplier::NaiveMatrixMultiplier(void)
{
}

Vector* NaiveMatrixMultiplier::vector_matrix_multiply(Vector* vector, Matrix* matrix)
{
	int m = matrix->height;
	int h = matrix->width;
	Vector* result = new Vector(h);
	for(int i = 0; i < h; i ++){
		float dot = 0;
		for (int j = 0; j < m; j ++){
			dot += vector->element_at(j) * matrix->element_at(j, i);
		}
		result->element_at(i) = dot;
	}
	return result;
}

NaiveMatrixMultiplier::~NaiveMatrixMultiplier(void)
{
}
