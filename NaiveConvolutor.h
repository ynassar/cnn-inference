#pragma once

#include "Matrix.h"

class NaiveConvolutor
{
public:
	static Matrix* convolute(Matrix* image, Matrix* filter);
	static void convolute_and_add(Matrix* image, Matrix* filter, Matrix* matrix_to_add_to);
	NaiveConvolutor(void);
	~NaiveConvolutor(void);
};

