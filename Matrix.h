#pragma once

#include "Vector.h"

class Matrix
{
public:
	float* data;
	int offset_h;
	int offset_w;
	int height, width;
	Matrix(void);
	Matrix(float* data, int height, int width);
	Matrix(int height, int width);
	Matrix(Vector*);
	Vector* flatten();
	float& element_at(int i, int j);
	void pad_zeros(int num_zeros_h, int num_zeros_w);
	~Matrix(void);
};

