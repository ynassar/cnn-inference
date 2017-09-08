#include "Matrix.h"
#include <cstring>
#include <iostream>


Matrix::Matrix(void)
{
}

Matrix::Matrix(Vector* vector){
	this->data = vector->data;
	this->height = 1;
	this->width = vector->size;
}

Matrix::Matrix(float* data, int height, int width){
	this->data = data;
	this->height = height;
	this->width = width;
}

Matrix::Matrix(int height, int width){
	this->height = height;
	this->width = width;
	this->data = new float[height*width];
	memset(this->data, 0, sizeof(float) * height * width);
}

Vector* Matrix::flatten(){
	return new Vector(this->data, this->height * this->width);
}

float& Matrix::element_at(int i, int j){
	if (i < this->offset_h || j < this->offset_w || i >= this->height - this->offset_h || j >= this->width - this->offset_w){
		float zero = 0;
		return zero;
	}

	return this->data[i * width + j];
}

void Matrix::pad_zeros(int num_zeros_h, int num_zeros_w)
{
	this->height += 2 * num_zeros_h;
	this->width += 2 * num_zeros_w;
	this->offset_h = num_zeros_h;
	this->offset_w = num_zeros_w;
}

Matrix::~Matrix(void)
{

}
