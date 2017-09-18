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
#include <cv.hpp>
#include "Utils.h"
#include <chrono>
using namespace std;
using namespace CNNInference;

const int NUM_RUNS = 100000;

int main(int argc, char** argv){
	string descriptor_file = argv[1];
	string image_file = argv[2];
	string mean_image_file = argv[3];
	Classifier* classifier = new Classifier(descriptor_file, mean_image_file);
	cout << "Predicting for image " << image_file << endl;
	Matrix<float>* predictions;
	cv::Mat raw_image = cv::imread(image_file, -1);
	cv::Mat converted;
	cv::Mat resized;
	cv::Mat floated;
	cv::cvtColor(raw_image, converted, cv::COLOR_BGR2GRAY);
	cv::resize(converted, resized, cv::Size(classifier->first_layer_input_height, classifier->first_layer_input_width));
	resized.convertTo(floated, CV_32FC1);
	Matrix<float>* img_mat = classifier->prepare_input((float*)floated.data);
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_RUNS; i++) {
		predictions = classifier->predict(img_mat);
	}
	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << NUM_RUNS << " predictions took " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() / (NUM_RUNS * 1.f) << " ns per run.\n";
	predictions->print_shape();
}