#pragma once

namespace CNNInference{
	class ThreeDimensionalArray
	{
	public:
		float* data;
		int depth, height, width;

		ThreeDimensionalArray(void);
		ThreeDimensionalArray(int depth, int height, int width);
		float& element_at(int i, int j, int k);
		~ThreeDimensionalArray(void);
	};
}
