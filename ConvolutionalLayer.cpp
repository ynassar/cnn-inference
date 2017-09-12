#include "ConvolutionalLayer.h"
#include "Mat.h"
#include <iostream>
#include "Mat.cpp"
using namespace std;
using namespace CNNInference;


ConvolutionalLayer::ConvolutionalLayer(ThreeDimensionalArray* filters, float* biases, int filter_size, int output_depth, int input_depth, int input_height, int input_width, int stride, int padding)
{
	this->raw_filters = filters;
	this->filter_size = filter_size;
	this->input_depth = input_depth;
	this->output_depth = output_depth;
	this->input_height = input_height;
	this->input_width = input_width;
	this->stride = stride;
	this->padding = padding;
	cout << "Entering im2col kernal filter_size = " << filter_size << endl;
	this->im2col_kernal(filters, input_depth, output_depth, filter_size);
	this->output_height = (this->input_height-this->filter_size)/this->stride +1;
	this->output_width = (this->input_width-this->filter_size)/this->stride +1;
	this->img_transformed = new Utils::Mat<float>(filter_size * filter_size * input_depth, output_height * output_width, 8);
	this->output = new Utils::Mat<float>(this->kernels->height, this->img_transformed->width, 8);
	for (int i = 0; i < this->kernels->height; i ++){
		for (int j = 0; j < this->img_transformed->width; j ++){
			(*this->output)[i][j] = 0;
		}
	}
	this->biases = new Utils::Mat<float>(*this->output);
	for (int i = 0; i < this->biases->height; i ++){
		for (int j = 0; j < this->biases->width; j ++){
			(*this->biases)[i][j] = biases[i];
		}
	}
}

Utils::Mat<float>* ConvolutionalLayer::forward(Utils::Mat<float>* input){
	im2col(*input);
	this->kernels->mult3(*this->img_transformed, this->output);
	this->output->element_wise_add_AVX(*this->biases);
	return this->output;
}

/*
 * for(int i=0;i<h;i+=stride)
		for(int j=0;j<w;j+=stride)
			for(int k=0;k<out_c;k++){
				float sum=0;
				for(int l=0;l<ker_size;l++)
					for(int m=0;m<ker_size;m++)
						for(int n=0;n<in_c;n++)
							sum+=kernal[k][l][m][n]*
							images[n].matrix[(i/stride+l)*images[n].phy_width+j/stride+m];
				res_conv[k][i][j] = sum;
			}
 */
//Utils::Mat<float>* ConvolutionalLayer::forward(Utils::Mat<float>* input){
//	for (int i = 0; i < this->input_height; i += this->stride){
//		for (int j = 0; j < this->input_width; j += this->stride){
//			for (int k = 0; k < this->output_depth; k ++){
//				float sum = 0;
//				for (int l = 0; l < this->filter_size; l ++){
//					for (int m = 0; m < this->filter_size; m ++){
//						for (int n = 0; n < this->input_depth; n ++){
//							sum += this->raw_filters[k].element_at(l,m,n) * (*input)[n][i*input_width+j+m];
//						}
//					}
//				}
//				(*output)[k][i*output_width+j] = sum;
//			}
//		}
//	}
//	return output;
//}

void ConvolutionalLayer::im2col_kernal(ThreeDimensionalArray* filters,
		int in_c,int out_c, int ker_size)
{
			this->kernels = new Utils::Mat<float> (out_c,ker_size*ker_size*in_c, 8);
			int count=0;
			for(int i=0;i<out_c;i++){
				count=0;
				for(int j=0;j<in_c;j++)
					for(int k=0;k<ker_size;k++)
						for(int l=0;l<ker_size;l++)
						{
							this->kernels->matrix[i*kernels->phy_width+count]= filters[i].element_at(j,k,l);//kernal[i][j][k][l];
							count++;
						}
			}
}
/*

Utils::Mat<float> im2col(Utils::Mat<float>& images, int f, int D0, int h, int w,int stride = 1){

	int h1 = (h-f)/stride +1;
	int w1 = (w-f)/stride +1;
	int len= D0*f*f*h1*w1;
	int p,q,d0,i,j;
	Utils::Mat<float> res(f*f*D0,h1*w1, 8);

	for(int k=0; k<len;k++)
	{
		p = k /(h1*w1);
		q = k %(h1*w1);
		d0 = (p/f)/f;
//			d0 = p/(h1*w1);
		i = (q/w1)+(p/f)%f;
		j = (q%w1)+ (p%f);
		cout<<p<<" "<<q<<" "<<i<<" "<<(p/f)%f<<" "<<j<<endl;
		res.matrix[p*res.phy_width+q] = images.matrix[d0*images.phy_width + i*w +j];
	}
//	res.print_shape();
	return res;
}
*/

void ConvolutionalLayer::im2col(Utils::Mat<float>& images){
	int h1 = this->output_height;
	int w1 = this->output_width;
	int len= this->input_depth*this->filter_size*this->filter_size*h1*w1;
	int p,q,d0,i,j;
	for(int k=0; k<len;k++)
	{
		p = k /(h1*w1);
		q = k %(h1*w1);
		d0 = (p/this->filter_size)/this->filter_size;
		i = (q/w1)+(p/this->filter_size)%this->filter_size;
		j = (q%w1)+ (p%this->filter_size);
		this->img_transformed->matrix[p*this->img_transformed->phy_width+q] = images[d0][i*this->input_width +j];
//		if (images[d0][i*this->input_width +j] == 0){
//			cout << "WTF" << endl;
//		}
	}

//	this->img_transformed->print_shape();
}

ConvolutionalLayer::~ConvolutionalLayer(void)
{
}
