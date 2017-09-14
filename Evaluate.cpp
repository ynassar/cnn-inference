/*
 * Evaluate.cpp
 *
 *  Created on: Sep 12, 2017
 *      Author: yousefnassar
 */
#include <iostream>
#include "Classifier.h"
#include "Matrix.h"
#include "Matrix.cpp"
#include "Utils.h"
#include <chrono>
using namespace std;
using namespace CNNInference;

const int NUM_RUNS = 100000;

int main(int argc, char** argv){
	string descriptor_file = argv[1];
	string image_file = argv[2];

	Classifier* classifier = new Classifier(descriptor_file);
	cout << "Predicting for image " << image_file << endl;
	Matrix<float>* img_mat = matrix_from_file(image_file);
	Matrix<float>* predictions;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_RUNS; i++) {
		predictions = classifier->predict(img_mat);
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << NUM_RUNS << " predictions took " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << " ns\n";
	predictions->print_shape();
}


