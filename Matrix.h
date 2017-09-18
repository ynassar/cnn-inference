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
	class Matrix {
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
		Matrix(const int height, const int width, int round);
		Matrix(const Matrix<T>& old);
		Matrix(std::vector<std::vector<T> >& nums);
		Matrix(const T** nums, const int& height, const int& width);
		~Matrix();
		void print_shape();
		void fill_rand(float bias);
		void fill_zeros();
		void transpose(Matrix<T>* res) const;
		void mult1(Matrix& B, Matrix* output);
		void mult2(const Matrix& op_B, Matrix* output);
		void mult3(const Matrix& op_B, Matrix* output);
		Matrix mult4(const Matrix& op_B);
		void mult5(Matrix& op_B, Matrix* output);
		Matrix& element_wise_mult(const Matrix& op_B) const;
		Matrix& avx_mult(const Matrix& op_B) const;
		Matrix element_wise_mult_AVX(const Matrix& op_B) const;
		Matrix element_wise_mult_SSE(const Matrix& op_B) const;
		Matrix& add(const Matrix& op_B) const;
		Matrix& avx_add(const Matrix& op_B) const;
		void element_wise_add_AVX(const Matrix& op_B);
		void element_wise_add_test(Matrix& op_B);
		T* operator [](const int index);

	};

} /* namespace Utils */

#endif /* MAT_H_ */
