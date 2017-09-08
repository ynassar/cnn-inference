#include "Classifier.h"
#include <iostream>
#include <time.h>
using namespace std;

int main(){
	srand(time(NULL));
	Classifier* classifier = new Classifier();
	ThreeDimensionalArray* data = new ThreeDimensionalArray(1, 1, 2);
	Vector* prediction = classifier->predict(data);
	for (int i = 0; i < prediction->size; ++i)
		cout << prediction->element_at(i) << endl;
}