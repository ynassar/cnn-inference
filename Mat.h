/*
 * Mat.h
 *
 *  Created on: Sep 5, 2017
 *      Author: ewais
 */

#ifndef MAT_H_
#define MAT_H_
#include<vector>

namespace CNNInference {

template<typename T>
class Mat {
private:

	void matmul_rec(const T *A, const T *B, T *C,
	int m, int n, int p, int fdA, int fdB, int fdC);
	void matmul_avx_8x8(const T *A, const T *B, T *C, int m,
			int n, int p, int fdA, int fdB, int fdC);
	void matmul_avx_4x4(const T *A, const T *B, T *C, int m,
			int n, int p, int fdA, int fdB, int fdC);
	void mult_4x4_trans(const T *A, const T *B, T *C,
			int phy_width, int phy_width_b);


public:
	T *matrix;
	int height, width, phy_height, phy_width, round;
	Mat(const int height, const int width, int round);
	Mat(const Mat<T>& old);
	Mat(std::vector<std::vector<T> >& nums);
	Mat(const T** nums, const int& height,const int& width);
	~Mat();
	void print_shape();
	void fill_rand(float bias);
	void fill_zeros();
	Mat transpose() const;
	Mat mult1(const Mat& op_B) const;
	void mult2(const Mat& op_B, Mat* output);
	void mult3(const Mat& op_B, Mat* output);
	Mat mult4(const Mat& op_B);
//	Mat mult5(Mat& op_B) const;
	Mat& element_wise_mult(const Mat& op_B) const;
	Mat& avx_mult(const Mat& op_B) const;
	Mat element_wise_mult_AVX(const Mat& op_B) const;
	Mat element_wise_mult_SSE(const Mat& op_B) const;
	Mat& add(const Mat& op_B) const;
	Mat& avx_add(const Mat& op_B) const;
	void element_wise_add_AVX(const Mat& op_B);
	T* operator [](const int index);

};

} /* namespace Utils */

#endif /* MAT_H_ */
