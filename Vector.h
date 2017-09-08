#pragma once

class Vector
{
public:
	float* data;
	int size;
	Vector(void);
	Vector(int size);
	float& element_at(int i);
	Vector(float* data, int size);
	~Vector(void);
};

