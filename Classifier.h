#pragma once

#include "Vector.h"
#include "ThreeDimensionalArray.h"
#include "Layer.h"
#include <vector>

class Classifier
{
	std::vector<Layer*> layers;
public:
	Classifier(void);
	~Classifier(void);

	Vector* predict(ThreeDimensionalArray* input);
};

