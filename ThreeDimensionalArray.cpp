#include "ThreeDimensionalArray.h"


ThreeDimensionalArray::ThreeDimensionalArray(void)
{
}

ThreeDimensionalArray::ThreeDimensionalArray(Vector* vector){
	this->matrices = new Matrix(vector);
	this->depth = 1;
}

ThreeDimensionalArray::ThreeDimensionalArray(Matrix* matrix){
	this->matrices = matrix;
	this->depth = 1;
}

ThreeDimensionalArray::ThreeDimensionalArray(Matrix* matrices, int depth){
	this->depth = depth;
	this->matrices = matrices;
}

Vector* ThreeDimensionalArray::to_vector(){
	return this->matrices[0].flatten();
}

Matrix* ThreeDimensionalArray::to_matrix(){
	return this->matrices;
}

float& ThreeDimensionalArray::element_at(int i, int j, int k){
	return this->matrices[i].element_at(j , k);
}

Matrix* ThreeDimensionalArray::matrix_at(int i){
	return this->matrices + i;
}

ThreeDimensionalArray::~ThreeDimensionalArray(void)
{
}
