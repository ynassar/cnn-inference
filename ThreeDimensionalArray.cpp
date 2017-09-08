#include "ThreeDimensionalArray.h"
#include <cstring>

ThreeDimensionalArray::ThreeDimensionalArray(void)
{
}

ThreeDimensionalArray::ThreeDimensionalArray(int depth, int height, int width){
	this->depth = depth;
	this->height = height;
	this->width = width;
	this->data = new float[depth*height*width];
	memset(this->data, 0, sizeof(float) * depth * height * width);
}

ThreeDimensionalArray::ThreeDimensionalArray(Vector* vector){
	this->data = vector->data;
	this->depth = 1;
	this->height = 1;
	this->width = vector->size;
}

ThreeDimensionalArray::ThreeDimensionalArray(Matrix* matrix){
	this->data = matrix->data;
	this->depth = 1;
	this->height = matrix->height;
	this->width = matrix->width;
}

Vector* ThreeDimensionalArray::to_vector(){
	return new Vector(this->data, this->width);
}

Matrix* ThreeDimensionalArray::to_matrix(){
	return new Matrix(this->data, this->height, this->width);
}

float& ThreeDimensionalArray::element_at(int i, int j, int k){
	return this->data[i * this->height * this->width + j * this->width + k];
}

Matrix* ThreeDimensionalArray::matrix_at(int i){
	return new Matrix(this->data + i * this->height * this->width, this->height, this->width);
}

ThreeDimensionalArray::~ThreeDimensionalArray(void)
{
}
