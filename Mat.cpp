/*
 * Mat.cpp
 *
 *  Created on: Sep 5, 2017
 *      Author: ewais
 */

#ifndef _MAT_CPP
#define _MAT_CPP

#include "Mat.h"
#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

inline float get_first( const __m128 vec){return _mm_cvtss_f32(_mm_shuffle_ps(vec,vec, _MM_SHUFFLE2(0,0)));}
namespace CNNInference {
template<typename T>
void mult_vector_elemnt_AVX(const T *A, const T *B, T* C, const int& len)
{
	__m256 ymm0,ymm1,ymm2;
	for(int i=0;i<len;i+=8)
	{
		ymm0 = _mm256_load_ps((float*)(B+i));
		ymm1 = _mm256_load_ps((float*)(A+i));
		ymm2 = _mm256_mul_ps(ymm0, ymm1);
		_mm256_store_ps((float *)(C+i), ymm2);
	}
}
template<typename T>
void add_vector_elemnt_AVX(const T *A, const T *B, T* C, const int& len)
{
	__m256 ymm0,ymm1,ymm2;
	for(int i=0;i<len;i+=8)
	{
		ymm0 = _mm256_load_ps((float*)(B+i));
		ymm1 = _mm256_load_ps((float*)(A+i));
		ymm2 = _mm256_add_ps(ymm0, ymm1);
		_mm256_store_ps((float *)(C+i), ymm2);
	}
}
template<typename T>
void mult_vector_elemnt_SSE(const T *A, const T *B, T* C, int len)
{
	__m128 ymm0,ymm1,ymm2;
	for(int i=0;i<len;i+=4)
	{
		ymm0 = _mm_load_ps((float*)(B+i));
		ymm1 = _mm_load_ps((float*)(A+i));
		ymm2 = _mm_mul_ps(ymm0, ymm1);
		_mm_store_ps((float *)(C+i), ymm2);
	}
}
template<typename T>
void mult_4x4(const T *A, const T *B, T *C, int phy_width, int phy_width_b)
{
	//Initialize some pointers
	const float * pMatrix2 = B;
	const float * pIn = A;
	float * pOut = C;

	__m128	ymm0, ymm1, ymm2, ymm3, ymm8, ymm9, ymm10, ymm11;
	ymm8 = _mm_load_ps((float *) (pMatrix2));
	ymm9 = _mm_load_ps((float *) (pMatrix2 + 1*phy_width_b));
	ymm10 = _mm_load_ps((float *) (pMatrix2 + 2*phy_width_b));
	ymm11 = _mm_load_ps((float *) (pMatrix2 + 3*phy_width_b));


	ymm0 = _mm_broadcast_ss(pIn);
	ymm1 = _mm_broadcast_ss(pIn + 1);
	ymm2 = _mm_broadcast_ss(pIn + 2);
	ymm3 = _mm_broadcast_ss(pIn + 3);

	ymm0 = _mm_mul_ps(ymm0, ymm8);
	ymm1 = _mm_mul_ps(ymm1, ymm9);
	ymm0 = _mm_add_ps(ymm0, ymm1);
	ymm2 = _mm_mul_ps(ymm2, ymm10);
	ymm3 = _mm_mul_ps(ymm3, ymm11);
	ymm2 = _mm_add_ps(ymm2, ymm3);
	ymm0 = _mm_add_ps(ymm0, ymm2);

	ymm2 = _mm_load_ps((float*)(pOut));
	ymm0 = _mm_add_ps(ymm0, ymm2);
	_mm_store_ps((float *) (pOut), ymm0);

	//Repeat using Matrix A Row 2
	ymm0 = _mm_broadcast_ss(pIn + 1*phy_width);
	ymm1 = _mm_broadcast_ss(pIn + 1*phy_width + 1);
	ymm2 = _mm_broadcast_ss(pIn + 1*phy_width + 2);
	ymm3 = _mm_broadcast_ss(pIn + 1*phy_width + 3);

	ymm0 = _mm_mul_ps(ymm0, ymm8);
	ymm1 = _mm_mul_ps(ymm1, ymm9);
	ymm0 = _mm_add_ps(ymm0, ymm1);
	ymm2 = _mm_mul_ps(ymm2, ymm10);
	ymm3 = _mm_mul_ps(ymm3, ymm11);
	ymm2 = _mm_add_ps(ymm2, ymm3);
	ymm0 = _mm_add_ps(ymm0, ymm2);

	ymm2 = _mm_load_ps((float*)(pOut+1*phy_width_b));
	ymm0 = _mm_add_ps(ymm0, ymm2);
	_mm_store_ps((float *) (pOut+1*phy_width_b), ymm0);

	//Repeat using Matrix A Row 3
	ymm0 = _mm_broadcast_ss(pIn + 2*phy_width);
	ymm1 = _mm_broadcast_ss(pIn + 2*phy_width + 1);
	ymm2 = _mm_broadcast_ss(pIn + 2*phy_width + 2);
	ymm3 = _mm_broadcast_ss(pIn + 2*phy_width + 3);

	ymm0 = _mm_mul_ps(ymm0, ymm8);
	ymm1 = _mm_mul_ps(ymm1, ymm9);
	ymm0 = _mm_add_ps(ymm0, ymm1);
	ymm2 = _mm_mul_ps(ymm2, ymm10);
	ymm3 = _mm_mul_ps(ymm3, ymm11);
	ymm2 = _mm_add_ps(ymm2, ymm3);
	ymm0 = _mm_add_ps(ymm0, ymm2);

	ymm2 = _mm_load_ps((float*)(pOut+2*phy_width_b));
	ymm0 = _mm_add_ps(ymm0, ymm2);
	_mm_store_ps((float *) (pOut+2*phy_width_b), ymm0);

	//Repeat using Matrix A Row 4
	ymm0 = _mm_broadcast_ss(pIn + 3*phy_width);
	ymm1 = _mm_broadcast_ss(pIn + 3*phy_width + 1);
	ymm2 = _mm_broadcast_ss(pIn + 3*phy_width + 2);
	ymm3 = _mm_broadcast_ss(pIn + 3*phy_width + 3);

	ymm0 = _mm_mul_ps(ymm0, ymm8);
	ymm1 = _mm_mul_ps(ymm1, ymm9);
	ymm0 = _mm_add_ps(ymm0, ymm1);
	ymm2 = _mm_mul_ps(ymm2, ymm10);
	ymm3 = _mm_mul_ps(ymm3, ymm11);
	ymm2 = _mm_add_ps(ymm2, ymm3);
	ymm0 = _mm_add_ps(ymm0, ymm2);

	ymm2 = _mm_load_ps((float*)(pOut+3*phy_width_b));
	ymm0 = _mm_add_ps(ymm0, ymm2);
	_mm_store_ps((float *) (pOut+3*phy_width_b), ymm0);
}
template<typename T>
void mult_8x8(const T *A, const T *B, T *C, int phy_width, int phy_width_b)
{
	__m256	ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
				ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

		//Initialize some pointers
		const float * pMatrix2 = B;
		const float * pIn = A;
		float * pOut = C;

		//Read the eight rows of Matrix B into ymm registers
		ymm8 = _mm256_load_ps((float *) (pMatrix2));
		ymm9 = _mm256_load_ps((float *) (pMatrix2 + 1*phy_width_b));
		ymm10 = _mm256_load_ps((float *) (pMatrix2 + 2*phy_width_b));
		ymm11 = _mm256_load_ps((float *) (pMatrix2 + 3*phy_width_b));
		ymm12 = _mm256_load_ps((float *) (pMatrix2 + 4*phy_width_b));
		ymm13 = _mm256_load_ps((float *) (pMatrix2 + 5*phy_width_b));
		ymm14 = _mm256_load_ps((float *) (pMatrix2 + 6*phy_width_b));
		ymm15 = _mm256_load_ps((float *) (pMatrix2 + 7*phy_width_b));

		//Broadcast each element of Matrix A Row 1 into a ymm register
		ymm0 = _mm256_broadcast_ss(pIn);
		ymm1 = _mm256_broadcast_ss(pIn + 1);
		ymm2 = _mm256_broadcast_ss(pIn + 2);
		ymm3 = _mm256_broadcast_ss(pIn + 3);
		ymm4 = _mm256_broadcast_ss(pIn + 4);
		ymm5 = _mm256_broadcast_ss(pIn + 5);
		ymm6 = _mm256_broadcast_ss(pIn + 6);
		ymm7 = _mm256_broadcast_ss(pIn + 7);

		//Multiply A11 times Row 1 of Matrix B
		ymm0 = _mm256_mul_ps(ymm0, ymm8);

		//Multiply A12 times Row 2 of Matrix B
		ymm1 = _mm256_mul_ps(ymm1, ymm9);

		//Create the first partial sum
		ymm0 = _mm256_add_ps(ymm0, ymm1);

		//Repeat for A13, A14, and Rows 3, 4 of Matrix B
		ymm2 = _mm256_mul_ps(ymm2, ymm10);
		ymm3 = _mm256_mul_ps(ymm3, ymm11);
		ymm2 = _mm256_add_ps(ymm2, ymm3);

		//Repeat for A15, A16, and Rows 5, 6 of Matrix B
		ymm4 = _mm256_mul_ps(ymm4, ymm12);
		ymm5 = _mm256_mul_ps(ymm5, ymm13);
		ymm4 = _mm256_add_ps(ymm4, ymm5);

		//Repeat for A17, A18, and Rows 7, 8 of Matrix B
		ymm6 = _mm256_mul_ps(ymm6, ymm14);
		ymm7 = _mm256_mul_ps(ymm7, ymm15);
		ymm6 = _mm256_add_ps(ymm6, ymm7);

		//Perform the final three adds
		ymm0 = _mm256_add_ps(ymm0, ymm2);
		ymm4 = _mm256_add_ps(ymm4, ymm6);
		ymm0 = _mm256_add_ps(ymm0, ymm4);

		//Store the result to Row 1 of Matrix C
		ymm4 = _mm256_load_ps((float*)(pOut));
		ymm0 = _mm256_add_ps(ymm0, ymm4);
		_mm256_store_ps((float *) (pOut), ymm0);

		//Repeat using Matrix A Row 2
		ymm0 = _mm256_broadcast_ss(pIn + 1*phy_width);
		ymm1 = _mm256_broadcast_ss(pIn + 1*phy_width + 1);
		ymm2 = _mm256_broadcast_ss(pIn + 1*phy_width + 2);
		ymm3 = _mm256_broadcast_ss(pIn + 1*phy_width + 3);
		ymm4 = _mm256_broadcast_ss(pIn + 1*phy_width + 4);
		ymm5 = _mm256_broadcast_ss(pIn + 1*phy_width + 5);
		ymm6 = _mm256_broadcast_ss(pIn + 1*phy_width + 6);
		ymm7 = _mm256_broadcast_ss(pIn + 1*phy_width + 7);
		ymm0 = _mm256_mul_ps(ymm0, ymm8);
		ymm1 = _mm256_mul_ps(ymm1, ymm9);
		ymm0 = _mm256_add_ps(ymm0, ymm1);
		ymm2 = _mm256_mul_ps(ymm2, ymm10);
		ymm3 = _mm256_mul_ps(ymm3, ymm11);
		ymm2 = _mm256_add_ps(ymm2, ymm3);
		ymm4 = _mm256_mul_ps(ymm4, ymm12);
		ymm5 = _mm256_mul_ps(ymm5, ymm13);
		ymm4 = _mm256_add_ps(ymm4, ymm5);
		ymm6 = _mm256_mul_ps(ymm6, ymm14);
		ymm7 = _mm256_mul_ps(ymm7, ymm15);
		ymm6 = _mm256_add_ps(ymm6, ymm7);
		ymm0 = _mm256_add_ps(ymm0, ymm2);
		ymm4 = _mm256_add_ps(ymm4, ymm6);
		ymm0 = _mm256_add_ps(ymm0, ymm4);

		ymm4 = _mm256_load_ps((float*)(pOut+phy_width_b));
		ymm0 = _mm256_add_ps(ymm0, ymm4);
		_mm256_store_ps((float *) (pOut+phy_width_b), ymm0);
		//Repeat using Matrix A Row 3
		ymm0 = _mm256_broadcast_ss(pIn + 2*phy_width);
		ymm1 = _mm256_broadcast_ss(pIn + 2*phy_width + 1);
		ymm2 = _mm256_broadcast_ss(pIn + 2*phy_width + 2);
		ymm3 = _mm256_broadcast_ss(pIn + 2*phy_width + 3);
		ymm4 = _mm256_broadcast_ss(pIn + 2*phy_width + 4);
		ymm5 = _mm256_broadcast_ss(pIn + 2*phy_width + 5);
		ymm6 = _mm256_broadcast_ss(pIn + 2*phy_width + 6);
		ymm7 = _mm256_broadcast_ss(pIn + 2*phy_width + 7);
		ymm0 = _mm256_mul_ps(ymm0, ymm8);
		ymm1 = _mm256_mul_ps(ymm1, ymm9);
		ymm0 = _mm256_add_ps(ymm0, ymm1);
		ymm2 = _mm256_mul_ps(ymm2, ymm10);
		ymm3 = _mm256_mul_ps(ymm3, ymm11);
		ymm2 = _mm256_add_ps(ymm2, ymm3);
		ymm4 = _mm256_mul_ps(ymm4, ymm12);
		ymm5 = _mm256_mul_ps(ymm5, ymm13);
		ymm4 = _mm256_add_ps(ymm4, ymm5);
		ymm6 = _mm256_mul_ps(ymm6, ymm14);
		ymm7 = _mm256_mul_ps(ymm7, ymm15);
		ymm6 = _mm256_add_ps(ymm6, ymm7);
		ymm0 = _mm256_add_ps(ymm0, ymm2);
		ymm4 = _mm256_add_ps(ymm4, ymm6);
		ymm0 = _mm256_add_ps(ymm0, ymm4);

		ymm4 = _mm256_load_ps((float*)(pOut+ 2*phy_width_b));
		ymm0 = _mm256_add_ps(ymm0, ymm4);
		_mm256_store_ps((float *) (pOut+ 2*phy_width_b), ymm0);
		//Repeat using Matrix A Row 4
		ymm0 = _mm256_broadcast_ss(pIn + 3*phy_width);
		ymm1 = _mm256_broadcast_ss(pIn + 3*phy_width + 1);
		ymm2 = _mm256_broadcast_ss(pIn + 3*phy_width + 2);
		ymm3 = _mm256_broadcast_ss(pIn + 3*phy_width + 3);
		ymm4 = _mm256_broadcast_ss(pIn + 3*phy_width + 4);
		ymm5 = _mm256_broadcast_ss(pIn + 3*phy_width + 5);
		ymm6 = _mm256_broadcast_ss(pIn + 3*phy_width + 6);
		ymm7 = _mm256_broadcast_ss(pIn + 3*phy_width + 7);
		ymm0 = _mm256_mul_ps(ymm0, ymm8);
		ymm1 = _mm256_mul_ps(ymm1, ymm9);
		ymm0 = _mm256_add_ps(ymm0, ymm1);
		ymm2 = _mm256_mul_ps(ymm2, ymm10);
		ymm3 = _mm256_mul_ps(ymm3, ymm11);
		ymm2 = _mm256_add_ps(ymm2, ymm3);
		ymm4 = _mm256_mul_ps(ymm4, ymm12);
		ymm5 = _mm256_mul_ps(ymm5, ymm13);
		ymm4 = _mm256_add_ps(ymm4, ymm5);
		ymm6 = _mm256_mul_ps(ymm6, ymm14);
		ymm7 = _mm256_mul_ps(ymm7, ymm15);
		ymm6 = _mm256_add_ps(ymm6, ymm7);
		ymm0 = _mm256_add_ps(ymm0, ymm2);
		ymm4 = _mm256_add_ps(ymm4, ymm6);
		ymm0 = _mm256_add_ps(ymm0, ymm4);

		ymm4 = _mm256_load_ps((float*)(pOut+ 3*phy_width_b));
		ymm0 = _mm256_add_ps(ymm0, ymm4);
		_mm256_store_ps((float *) (pOut+ 3*phy_width_b), ymm0);
		//Repeat using Matrix A Row 5
		ymm0 = _mm256_broadcast_ss(pIn + 4*phy_width);
		ymm1 = _mm256_broadcast_ss(pIn + 4*phy_width + 1);
		ymm2 = _mm256_broadcast_ss(pIn + 4*phy_width + 2);
		ymm3 = _mm256_broadcast_ss(pIn + 4*phy_width + 3);
		ymm4 = _mm256_broadcast_ss(pIn + 4*phy_width + 4);
		ymm5 = _mm256_broadcast_ss(pIn + 4*phy_width + 5);
		ymm6 = _mm256_broadcast_ss(pIn + 4*phy_width + 6);
		ymm7 = _mm256_broadcast_ss(pIn + 4*phy_width + 7);
		ymm0 = _mm256_mul_ps(ymm0, ymm8);
		ymm1 = _mm256_mul_ps(ymm1, ymm9);
		ymm0 = _mm256_add_ps(ymm0, ymm1);
		ymm2 = _mm256_mul_ps(ymm2, ymm10);
		ymm3 = _mm256_mul_ps(ymm3, ymm11);
		ymm2 = _mm256_add_ps(ymm2, ymm3);
		ymm4 = _mm256_mul_ps(ymm4, ymm12);
		ymm5 = _mm256_mul_ps(ymm5, ymm13);
		ymm4 = _mm256_add_ps(ymm4, ymm5);
		ymm6 = _mm256_mul_ps(ymm6, ymm14);
		ymm7 = _mm256_mul_ps(ymm7, ymm15);
		ymm6 = _mm256_add_ps(ymm6, ymm7);
		ymm0 = _mm256_add_ps(ymm0, ymm2);
		ymm4 = _mm256_add_ps(ymm4, ymm6);
		ymm0 = _mm256_add_ps(ymm0, ymm4);

		ymm4 = _mm256_load_ps((float*)(pOut+ 4*phy_width_b));
		ymm0 = _mm256_add_ps(ymm0, ymm4);
		_mm256_store_ps((float *) (pOut+ 4*phy_width_b), ymm0);

		//Repeat using Matrix A Row 6
		ymm0 = _mm256_broadcast_ss(pIn + 5*phy_width);
		ymm1 = _mm256_broadcast_ss(pIn + 5*phy_width + 1);
		ymm2 = _mm256_broadcast_ss(pIn + 5*phy_width + 2);
		ymm3 = _mm256_broadcast_ss(pIn + 5*phy_width + 3);
		ymm4 = _mm256_broadcast_ss(pIn + 5*phy_width + 4);
		ymm5 = _mm256_broadcast_ss(pIn + 5*phy_width + 5);
		ymm6 = _mm256_broadcast_ss(pIn + 5*phy_width + 6);
		ymm7 = _mm256_broadcast_ss(pIn + 5*phy_width + 7);
		ymm0 = _mm256_mul_ps(ymm0, ymm8);
		ymm1 = _mm256_mul_ps(ymm1, ymm9);
		ymm0 = _mm256_add_ps(ymm0, ymm1);
		ymm2 = _mm256_mul_ps(ymm2, ymm10);
		ymm3 = _mm256_mul_ps(ymm3, ymm11);
		ymm2 = _mm256_add_ps(ymm2, ymm3);
		ymm4 = _mm256_mul_ps(ymm4, ymm12);
		ymm5 = _mm256_mul_ps(ymm5, ymm13);
		ymm4 = _mm256_add_ps(ymm4, ymm5);
		ymm6 = _mm256_mul_ps(ymm6, ymm14);
		ymm7 = _mm256_mul_ps(ymm7, ymm15);
		ymm6 = _mm256_add_ps(ymm6, ymm7);
		ymm0 = _mm256_add_ps(ymm0, ymm2);
		ymm4 = _mm256_add_ps(ymm4, ymm6);
		ymm0 = _mm256_add_ps(ymm0, ymm4);

		ymm4 = _mm256_load_ps((float*)(pOut+ 5*phy_width_b));
		ymm0 = _mm256_add_ps(ymm0, ymm4);
		_mm256_store_ps((float *) (pOut+ 5*phy_width_b), ymm0);

		//Repeat using Matrix A Row 7
		ymm0 = _mm256_broadcast_ss(pIn + 6*phy_width);
		ymm1 = _mm256_broadcast_ss(pIn + 6*phy_width + 1);
		ymm2 = _mm256_broadcast_ss(pIn + 6*phy_width + 2);
		ymm3 = _mm256_broadcast_ss(pIn + 6*phy_width + 3);
		ymm4 = _mm256_broadcast_ss(pIn + 6*phy_width + 4);
		ymm5 = _mm256_broadcast_ss(pIn + 6*phy_width + 5);
		ymm6 = _mm256_broadcast_ss(pIn + 6*phy_width + 6);
		ymm7 = _mm256_broadcast_ss(pIn + 6*phy_width + 7);
		ymm0 = _mm256_mul_ps(ymm0, ymm8);
		ymm1 = _mm256_mul_ps(ymm1, ymm9);
		ymm0 = _mm256_add_ps(ymm0, ymm1);
		ymm2 = _mm256_mul_ps(ymm2, ymm10);
		ymm3 = _mm256_mul_ps(ymm3, ymm11);
		ymm2 = _mm256_add_ps(ymm2, ymm3);
		ymm4 = _mm256_mul_ps(ymm4, ymm12);
		ymm5 = _mm256_mul_ps(ymm5, ymm13);
		ymm4 = _mm256_add_ps(ymm4, ymm5);
		ymm6 = _mm256_mul_ps(ymm6, ymm14);
		ymm7 = _mm256_mul_ps(ymm7, ymm15);
		ymm6 = _mm256_add_ps(ymm6, ymm7);
		ymm0 = _mm256_add_ps(ymm0, ymm2);
		ymm4 = _mm256_add_ps(ymm4, ymm6);
		ymm0 = _mm256_add_ps(ymm0, ymm4);

		ymm4 = _mm256_load_ps((float*)(pOut+ 6 * phy_width_b));
		ymm0 = _mm256_add_ps(ymm0, ymm4);
		_mm256_store_ps((float *) (pOut+ 6 * phy_width_b), ymm0);

		//Repeat using Matrix A Row 8
		ymm0 = _mm256_broadcast_ss(pIn + 7*phy_width);
		ymm1 = _mm256_broadcast_ss(pIn + 7*phy_width + 1);
		ymm2 = _mm256_broadcast_ss(pIn + 7*phy_width + 2);
		ymm3 = _mm256_broadcast_ss(pIn + 7*phy_width + 3);
		ymm4 = _mm256_broadcast_ss(pIn + 7*phy_width + 4);
		ymm5 = _mm256_broadcast_ss(pIn + 7*phy_width + 5);
		ymm6 = _mm256_broadcast_ss(pIn + 7*phy_width + 6);
		ymm7 = _mm256_broadcast_ss(pIn + 7*phy_width + 7);
		ymm0 = _mm256_mul_ps(ymm0, ymm8);
		ymm1 = _mm256_mul_ps(ymm1, ymm9);
		ymm0 = _mm256_add_ps(ymm0, ymm1);
		ymm2 = _mm256_mul_ps(ymm2, ymm10);
		ymm3 = _mm256_mul_ps(ymm3, ymm11);
		ymm2 = _mm256_add_ps(ymm2, ymm3);
		ymm4 = _mm256_mul_ps(ymm4, ymm12);
		ymm5 = _mm256_mul_ps(ymm5, ymm13);
		ymm4 = _mm256_add_ps(ymm4, ymm5);
		ymm6 = _mm256_mul_ps(ymm6, ymm14);
		ymm7 = _mm256_mul_ps(ymm7, ymm15);
		ymm6 = _mm256_add_ps(ymm6, ymm7);
		ymm0 = _mm256_add_ps(ymm0, ymm2);
		ymm4 = _mm256_add_ps(ymm4, ymm6);
		ymm0 = _mm256_add_ps(ymm0, ymm4);

		ymm4 = _mm256_load_ps((float*)(pOut+ 7 * phy_width_b));
		ymm0 = _mm256_add_ps(ymm0, ymm4);
		_mm256_store_ps((float*)(pOut+ 7 * phy_width_b), ymm0);
}
template<typename T>
inline void transpose4x4_SSE(const T *A, T *B, const int lda, const int ldb) {
    __m128 row1 = _mm_load_ps(&A[0]);
    __m128 row2 = _mm_load_ps(&A[lda]);
    __m128 row3 = _mm_load_ps(&A[2*lda]);
    __m128 row4 = _mm_load_ps(&A[3*lda]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[0], row1);
     _mm_store_ps(&B[ldb], row2);
     _mm_store_ps(&B[2*ldb], row3);
     _mm_store_ps(&B[3*ldb], row4);
}
template<typename T>
inline void transpose_block_SSE4x4(const T *A, T *B, const int n, const int m,
		const int lda, const int ldb ,const int block_size) {
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            int max_i2 = i+block_size < n ? i + block_size : n;
            int max_j2 = j+block_size < m ? j + block_size : m;
            for(int i2=i; i2<max_i2; i2+=4) {
                for(int j2=j; j2<max_j2; j2+=4) {
                    transpose4x4_SSE(&A[i2*lda +j2], &B[j2*ldb + i2], lda, ldb);
                }
            }
        }
    }
}
template<typename T>
void Mat<T>::mult_4x4_trans(const T *A, const T *B, T *C, int phy_width, int phy_width_b){
	__m128	ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
					ymm8, ymm9, ymm10, ymm11, ymm12,ymm13,ymm14,ymm15;


	ymm0 = _mm_load_ps((float *) (B));
	ymm1 = _mm_load_ps((float *) (B+phy_width_b));
	ymm2 = _mm_load_ps((float *) (B+2*phy_width_b));
	ymm3 = _mm_load_ps((float *) (B+3*phy_width_b));

	ymm4 = _mm_load_ps((float *) (A));
	ymm5 = _mm_load_ps((float *) (A+phy_width));
	ymm6 = _mm_load_ps((float *) (A+2*phy_width));
	ymm7 = _mm_load_ps((float *) (A+3*phy_width));

	ymm12 = _mm_load_ps((float*)(C));
	ymm13 = _mm_load_ps((float*)(C+phy_width_b));
	ymm14 = _mm_load_ps((float*)(C+2*phy_width_b));
	ymm15 = _mm_load_ps((float*)(C+3*phy_width_b));

	ymm8 = _mm_dp_ps(ymm4,ymm0,0xff);
	ymm9 = _mm_dp_ps(ymm4,ymm1,0xff);
	ymm10 = _mm_dp_ps(ymm4,ymm2,0xff);
	ymm11 = _mm_dp_ps(ymm4,ymm3,0xff);


	ymm4 = _mm_set_ps(get_first(ymm11),get_first(ymm10),get_first(ymm9),get_first(ymm8));
	ymm12 = _mm_add_ps(ymm12,ymm4);
	_mm_store_ps(C, ymm12);

	ymm8 = _mm_dp_ps(ymm5,ymm0,0xff);
	ymm9 = _mm_dp_ps(ymm5,ymm1,0xff);
	ymm10 = _mm_dp_ps(ymm5,ymm2,0xff);
	ymm11 = _mm_dp_ps(ymm5,ymm3,0xff);

	ymm4 = _mm_set_ps(get_first(ymm11),get_first(ymm10),get_first(ymm9),get_first(ymm8));
	ymm13 = _mm_add_ps(ymm13,ymm4);
	_mm_store_ps(C+phy_width_b, ymm13);


	ymm8 = _mm_dp_ps(ymm6,ymm0,0xff);
	ymm9 = _mm_dp_ps(ymm6,ymm1,0xff);
	ymm10 = _mm_dp_ps(ymm6,ymm2,0xff);
	ymm11 = _mm_dp_ps(ymm6,ymm3,0xff);


	ymm4 = _mm_set_ps(get_first(ymm11),get_first(ymm10),get_first(ymm9),get_first(ymm8));
	ymm14 = _mm_add_ps(ymm14,ymm4);
	_mm_store_ps(C+2*phy_width_b, ymm14);




	ymm8 = _mm_dp_ps(ymm7,ymm0,0xff);
	ymm9 = _mm_dp_ps(ymm7,ymm1,0xff);
	ymm10 = _mm_dp_ps(ymm7,ymm2,0xff);
	ymm11 = _mm_dp_ps(ymm7,ymm3,0xff);

	ymm4 = _mm_set_ps(get_first(ymm11),get_first(ymm10),get_first(ymm9),get_first(ymm8));
	ymm15 = _mm_add_ps(ymm15,ymm4);
	_mm_store_ps(C+3*phy_width_b, ymm15);
}

template<typename T>
void Mat<T>::matmul_avx_8x8(const T *A, const T *B, T *C, int m, int n, int p, int fdA, int fdB, int fdC){
	for(int i=0;i<m;i+=8){//for A rows
		for(int j=0;j<n;j+=8){//for A cols, B rows
			for(int k=0;k<p;k+=8){ //for B cols
				mult_8x8(A+i*fdA+j, B+j*fdB+k, C+i*fdC+k, fdA, fdB);
			}
		}
	}
}
template<typename T>
void Mat<T>::matmul_avx_4x4(const T *A, const T *B, T *C, int m, int n, int p, int fdA, int fdB, int fdC){
	for(int i=0;i<m;i+=4){//for A rows
		for(int j=0;j<n;j+=4){//for A cols, B rows
			for(int k=0;k<p;k+=4){ //for B cols
				mult_4x4(A+i*fdA+j, B+j*fdB+k, C+i*fdC+k, fdA, fdB);
			}
		}
	}
}
template<typename T>
void Mat<T>::matmul_rec(const T *A, const T *B, T *C,
int m, int n, int p, int fdA, int fdB, int fdC)
{
 if (m+n+p <= 48) { /* <= 16x16 matrices "on average" */
 int i, j, k;
 for (i = 0; i < m; ++i){
	 int ai= i*fdA;
	 for (k = 0; k < p; ++k) {
		 float sum = 0.1f;
		 for (j = 0; j < n; ++j)
			 sum += A[ai+j] * B[j*fdB + k];
		 C[i*fdC + k] += sum;
	 	 }
	 }
 }
 else { /* divide and conquer */
 int m2 = m/2, n2 = n/2, p2 = p/2;
 matmul_rec(A, B, C, m2, n2, p2, fdA, fdB, fdC);
 matmul_rec(A+n2, B+n2*fdB, C, m2, n-n2, p2, fdA, fdB, fdC);
 matmul_rec(A, B+p2, C+p2, m2, n2, p-p2, fdA, fdB, fdC);
 matmul_rec(A+n2, B+p2+n2*fdB, C, m2, n-n2, p-p2, fdA, fdB, fdC);
 matmul_rec(A+m2*fdA, B, C+m2*fdC, m-m2, n2, p2, fdA, fdB, fdC);
 matmul_rec(A+m2*fdA+n2, B+n2*fdB, C+m2*fdC, m-m2, n-n2, p2, fdA, fdB, fdC);
 matmul_rec(A+m2*fdA, B+p2, C+m2*fdC+p2, m-m2, n2, p-p2, fdA, fdB, fdC);
 matmul_rec(A+m2*fdA+n2, B+p2+n2*fdB, C+m2*fdC, m-m2, n-n2, p-p2, fdA, fdB, fdC);
 }
}

template<typename T>
	Mat<T>::Mat(const int h, const int w, int r):height(h), width(w),round(r){
	int align = 64;
	if(r==4)
		align = 32;
	#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
	phy_height = ROUND_UP(h, round);
	phy_width = ROUND_UP(w, round);
	matrix = (T*)_mm_malloc(sizeof(T)*phy_height*phy_width, align);
	}

template<typename T>
	Mat<T>::Mat(const Mat<T>& x):height(x.height), width(x.width),round(x.round),
	phy_width(x.phy_width), phy_height(x.phy_height){
	int align = 32;
	if(round==4)
		align = 16;
//		std::cout<<phy_height<<" "<<phy_width<<std::endl;
//	std::cout<<align<<std::endl;
	matrix = (T*)_mm_malloc(sizeof(T)*phy_height*phy_width, 64);
		for(int i=0;i<phy_height*phy_width;i++)
		{
			matrix[i] = x.matrix[i];
		}
	}
	template<typename T>
	Mat<T>::~Mat(){
//		std::cout<<"ermy"<<std::endl;
		_mm_free((void*)matrix);
	}


	template<typename T>
	void Mat<T>::print_shape(){
		for(int i=0;i<this->height;i++){
			int ai = i*phy_width;
			for(int j=0;j<this->width;j++){
				std::cout<<matrix[ai+j]<<" ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}

	template<typename T>
	void Mat<T>::fill_rand(float bias){
		for(int i=0;i<this->height;i++){
			int ai = i*phy_width;
			for(int j=0;j<this->width;j++)
				matrix[ai+j] = bias+(float(rand() / float(0x3f3f3f3f)));
		}

	}

	template<typename T>
	void Mat<T>::fill_zeros(){
		for(int i=0;i<this->height;i++){
			int ai = i*phy_width;
			for(int j=0;j<this->width;j++)
				matrix[ai+j] = 0.0000001;
		}

	}



	template<typename T>
	Mat<T> Mat<T>::transpose() const{
		Mat<T> res(this->width, this->height, round);
		for(int i=0;i<this->height;i++)
		{
			int ai = i*phy_width;
			for(int j=0;j<this->width;j++)
			{
				res[j][i]= matrix[ai+j];
			}
		}
		return res;
	}

	template<typename T>
	T* Mat<T>::operator [](const int index){
		return matrix+index*phy_width;
	}


	template<typename T>
		Mat<T> Mat<T>:: mult1(const Mat<T>& B) const{
			Mat<T> op_B = B.transpose();
			Mat<T> res(this->height, op_B.height, round);

			int h = this->height, w = this->width;
			for(int i=0;i<h;i++){
				int ai = i*phy_width;
				for(int j=0;j<op_B.height;j++){
					T* bi= op_B[j];
					T temp = 0.0000001;
					for(int k=0; k<w;k++){
						temp+= matrix[ai+k]*bi[k];
					}
					res[i][j]=temp;
				}
			}
			return res;
		}

	template<typename T>
	void Mat<T>:: mult2(const Mat<T>& B, Mat* output){
		memset(output->matrix, 0, sizeof(float) * output->phy_height*output->phy_width);
		matmul_rec(this->matrix, B.matrix, output->matrix, this->height, this->width,
				B.width, this->phy_width, B.phy_width, output->phy_width);
	}
	template<typename T>
	void Mat<T>:: mult3(const Mat<T>& B, Mat* output){
		memset(output->matrix, 0, sizeof(float) * output->phy_height*output->phy_width);
		matmul_avx_8x8(this->matrix, B.matrix, output->matrix, this->phy_height, this->phy_width,
				B.phy_width, this->phy_width, B.phy_width, B.phy_width);
	}
	template<typename T>
	Mat<T> Mat<T>:: mult4(const Mat<T>& B){
		Mat<T> res(this->height, B.width, 4);
		memset(res.matrix, 0, sizeof(float) * res.phy_height * res.phy_width);
		matmul_avx_4x4(this->matrix, B.matrix, res.matrix, this->phy_height, this->phy_width,
				B.phy_width, this->phy_width, B.phy_width, B.phy_width);
		return res;
	}
	template<typename T>
	Mat<T> Mat<T>::element_wise_mult_SSE(const Mat& op_B) const{
		Mat<T> res(height, width, round);
		for(int i=0;i<phy_height;i++)
			mult_vector_elemnt_SSE(matrix+i*phy_width, op_B.matrix+i*op_B.phy_width,
					res.matrix+i*phy_width, phy_width);
		return res;
	}
	template<typename T>
	Mat<T> Mat<T>::element_wise_mult_AVX(const Mat& op_B) const{
		Mat<T> res(height, width, round);
		for(int i=0;i<phy_height;i++)
			mult_vector_elemnt_AVX(matrix+i*phy_width, op_B.matrix+i*op_B.phy_width,
					res.matrix+i*phy_width, phy_width);
		return res;
	}
	template<typename T>
	void Mat<T>::element_wise_add_AVX(const Mat& op_B){
		for(int i=0;i<phy_height;i++)
			add_vector_elemnt_AVX(matrix+i*phy_width, op_B.matrix+i*op_B.phy_width,
					this->matrix+i*phy_width, phy_width);
	}

}


#endif
