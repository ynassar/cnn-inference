#include "Vector.h"
#include <iostream>
#include <cstring>


Vector::Vector(void)
{
}

Vector::Vector(int size){
	this->size = size;
	this->data = new float[size];
	for(int i = 0; i < size; ++i){
		this->data[i] = 1;
	}
	//memset(this->data, 0, sizeof(float) * size);
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
