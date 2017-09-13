#pragma once

#include "Matrix.h"

namespace CNNInference {

	Matrix<float>* MatrixFromFile(const std::string& txtfile);
}