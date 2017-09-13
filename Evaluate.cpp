/*
 * Evaluate.cpp
 *
 *  Created on: Sep 12, 2017
 *      Author: yousefnassar
 */
#include <iostream>
#include "Classifier.h"
#include "Mat.h"
#include "Mat.cpp"
using namespace std;
using namespace CNNInference;

int main(int argc, char** argv){
	string descriptor_file = argv[1];
	string image_file = argv[2];

	Classifier* classifier = new Classifier(descriptor_file);

	cout << "Predicting for image " << image_file << endl;
	CNNInference::Matrix<float>* img_mat = classifier->MatrixFromFile(image_file);
	CNNInference::Matrix<float>* predictions = classifier->predict(img_mat);
	predictions->print_shape();
}



