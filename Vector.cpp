#include "Vector.h"


Vector::Vector(void)
{
}

Vector::Vector(int size){
	this->size = size;
	this->data = new float[size];
}

Vector::Vector(float* data, int size){
	this->data = data;
	this->size = size;
}

float& Vector::element_at(int i){
	return this->data[i];
}

Vector::~Vector(void)
{
}
