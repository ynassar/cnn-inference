#include "Utils.h"
#include <fstream>
#include <string>

namespace CNNInference {
	Matrix<float>* MatrixFromFile(const std::string& txtfile) {
		std::ifstream infile(txtfile);
		int channels_in, height, width;
		infile >> channels_in >> height >> width;
		Matrix<float>* input_mat = new Matrix<float>(channels_in, height * width, 1);
		int temp;
		float* data = new float[channels_in * height * width];
		for (int i = 0; i < channels_in * height * width; i++) {
			infile >> data[i];
		}

		for (int i = 0; i < channels_in; i++) {
			for (int j = 0; j < height; j++) {
				for (int k = 0; k < width; k++) {
					input_mat->operator [](i)[j*width + k] = data[i * width * height + j * width + k];
				}
			}
		}
		return input_mat;
	}
}