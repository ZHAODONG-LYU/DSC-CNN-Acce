#include <stdint.h>
#include <stdlib.h>
#include "dwcnn_quantized_opt.h"
#include "math.h"
#include <fstream>
#include <iostream>
#include "ap_fixed.h"	//ap_fixed<18,6,AP_TRN_ZERO,AP_SAT>		<W,I,Q,O,N>
#include "ap_int.h"	//ap_int<N> or ap_uint<N>, 1<=N<=1024
#include "hls_stream.h"
using namespace hls;


static ap_int<EIGHTBITS> depthwisekernel_opt[16] = {
		-48,-9,-22,-25,-11,13,40,24,-2,23,10,-72,-30,-33,61,70
};
static ap_int<EIGHTBITS> pointwisekernel_opt[64] = {
		-57,16,22,-11,9,17,61,-51,60,-30,-16,-36,15,-38,7,57,-58,23,42,58,-54,55,-9,-3,-26,-60,-26,63,24,-41,-47,21,-34,33,21,-37,-23,-13,1,-2,-42,-10,-12,6,32,47,-44,-22,-45,-18,-67,60,25,-25,-10,21,-20,0,-70,-32,-4,-54,39,26
};
static ap_int<EIGHTBITS> SeparateBias_opt[16] = {
		-96,22,-41,-2,13,-7,-32,-16,-60,-67,-88,-48,3,-23,-14,19
};
static ap_int<EIGHTBITS> BatchN_Scale_opt[16] = {
		127,124,127,127,126,127,127,127,127,127,127,127,121,127,126,127
};
static ap_int<EIGHTBITS> BatchN_Offset_opt[16] = {
		-19,47,-17,1,-68,-57,3,29,4,-32,-7,44,-18,18,-36,-25
};
static ap_int<EIGHTBITS> BatchN_tMean_opt[16] = {
		63,7,1,42,127,122,98,10,21,9,41,3,127,6,15,73
};
static ap_int<EIGHTBITS> BatchN_tVariance_opt[16] = {
		19,2,1,33,90,43,54,8,6,1,9,1,68,4,3,27
};
static ap_int<EIGHTBITS> Fenmu_opt[16] = {
		49,16,12,65,107,74,83,32,28,12,34,12,93,22,20,58
};
static ap_int<EIGHTBITS> Conv2_Weights_opt[1024] = {
		-16,14,12,13,16,12,-3,-11,17,-21,-26,-6,18,-24,8,25,-18,38,18,15,4,-9,30,12,-17,-45,13,3,27,-14,7,8,5,8,-27,12,-9,14,20,-7,-3,12,13,7,11,13,0,41,-1,14,17,16,29,-27,0,14,5,2,-7,-24,-2,-14,34,14,11,-22,14,-4,7,-11,4,-14,3,35,9,-34,-9,-7,23,35,16,-19,1,16,29,-14,2,3,-27,33,8,-5,29,-5,25,27,29,-1,15,-15,11,33,14,-25,21,25,-10,-33,26,-23,27,21,0,9,0,-70,-15,14,-36,-26,14,29,1,-41,-36,-27,33,-30,-8,-31,2,-12,-59,-19,-14,-37,-2,-24,-17,-6,-31,-2,-7,-9,-16,-28,-19,-14,-36,-25,-20,-29,-24,0,-44,-10,-22,9,4,-5,-34,-13,-6,-17,-31,-64,-25,-34,-25,-18,-1,13,-41,11,-8,11,-30,-21,-3,10,-46,-28,-16,-10,-14,11,-29,12,-18,24,-1,-6,-49,-20,20,4,23,10,27,-13,-32,-24,-2,7,-1,20,35,34,-20,5,46,37,1,-12,1,6,-37,-18,-28,15,-13,-3,2,12,-46,-15,21,-36,3,20,9,-26,-30,-30,-5,-19,-7,-48,26,-9,-24,-31,8,-44,24,40,-11,-4,-22,-22,5,-25,26,-43,28,-6,5,-38,24,11,-7,11,-12,21,10,-18,3,-45,1,52,16,-13,3,3,54,27,5,-1,-12,55,-2,-17,-15,15,-4,32,12,10,-5,-7,39,14,-1,-4,5,14,-7,-13,-18,7,7,-37,19,10,-30,4,40,36,-18,-27,0,16,-17,-7,-23,-7,2,22,-9,10,34,-24,-18,-16,-13,18,0,-47,18,54,53,-19,17,-32,-2,17,-15,-20,4,-30,-47,-18,-41,-28,-3,-1,-14,-18,-42,0,16,-34,-31,-11,-20,-22,-21,-2,-21,-40,-32,-3,5,-11,-32,2,-14,-16,-33,-22,19,1,-5,-17,-26,10,-27,29,-9,-3,-26,17,2,6,34,-3,6,5,18,22,10,-36,-6,29,40,-64,30,-39,-35,18,25,-17,5,-23,7,13,-7,-22,37,15,29,-74,3,-20,-12,-4,-13,-10,-5,-44,-21,8,-31,-18,-2,20,-29,-40,-23,-17,3,-26,-37,-17,6,-24,-16,2,-41,-25,-30,-32,-57,-11,-53,-20,-28,-46,-27,-10,6,25,-9,3,-20,-41,-38,-39,-39,-30,-8,-19,8,14,-62,-18,5,14,1,0,-14,-22,-10,-45,-39,-35,12,-2,-7,15,-33,-14,-15,16,3,-3,4,-32,-5,-52,-9,-11,-28,0,8,4,-55,5,4,3,28,-24,-9,-35,-38,-47,-17,-29,-15,5,10,-2,-3,-29,12,2,17,13,-19,42,21,-21,-12,-35,15,27,31,7,-18,5,65,30,10,-13,3,52,5,-20,-22,17,15,22,14,11,-12,18,5,25,-22,-21,5,29,-19,-19,-9,10,3,33,12,22,-26,-26,-12,-7,-15,-1,-7,40,-9,-20,-27,-14,-2,19,-13,19,-3,-42,-3,8,15,13,12,14,-25,-14,11,-46,23,-8,-18,44,2,-13,11,8,4,-10,23,3,-38,-32,3,0,19,0,-24,52,-15,-17,14,-13,8,-16,37,-3,-38,-6,-9,23,6,26,-6,36,-4,-5,32,-24,2,-7,38,-6,-43,-2,7,-7,22,-28,12,22,12,13,38,-25,26,12,20,14,20,2,-10,4,0,46,20,9,-18,-8,-12,27,30,17,-3,43,0,-2,-25,-7,13,-23,23,18,-8,-26,45,7,15,-20,1,35,-30,-1,-5,10,-5,4,-9,42,-24,-18,28,-6,0,-25,4,17,-10,1,-19,-17,-13,10,-12,14,-17,6,26,-1,-3,7,15,35,-10,-1,-4,12,10,40,20,-5,-3,-20,3,-15,9,20,1,2,-31,5,-17,21,14,-57,0,0,-33,-21,-32,-5,3,26,8,-36,-28,-9,0,-9,2,-44,11,4,-28,-35,28,13,-13,-8,6,31,-33,-3,-12,-35,-7,-26,28,30,8,-34,-16,0,22,29,-17,-44,-14,-4,-26,-18,0,0,29,-26,-50,-24,14,10,12,20,19,42,-32,6,-18,-31,14,16,17,-20,-31,6,40,2,23,11,-7,22,-17,-5,-18,19,0,14,26,-15,-25,-38,29,7,14,6,6,-27,-34,-9,-2,0,10,43,13,-23,20,-17,7,-2,32,18,17,-20,0,40,38,-63,15,-22,-9,24,22,-4,8,-5,24,59,1,-15,30,20,1,-56,7,-35,-2,-5,-4,-8,3,-54,-26,11,-57,-27,11,-14,-31,-37,-22,-23,13,-56,-19,-20,-10,-36,-14,-29,-38,-24,-11,29,-39,-14,-24,-24,-28,-14,4,8,21,6,12,-24,26,31,19,-35,-24,-16,20,32,-18,10,13,-2,44,16,-14,15,2,28,-4,-26,-4,25,12,25,-8,0,4,-4,13,-26,26,19,5,-9,16,-21,-17,-4,10,-16,-13,19,2,-17,-5,34,-14,1,8,-4,-5,-26,-22,4,5,15,-10,41,1,-34,20,14,28,27,10,-8,0,11,14,-36,11,-4,18,26,-8,-17,4,12,28,24,17,-15,-19,43,5,-16,-3,5,24,18,20,5,-2,-1,20,19,7,-36,14,44,11,-30,-4,-26,21,5,30,5,3,-50,-39,9,-30,-31,10,36,-21,-20,-18,-31,18,-10
};
static ap_int<EIGHTBITS> Conv2_Bias_opt[16] = {
		-17,-49,-6,-24,-15,-19,-57,-29,-38,-14,-20,13,-27,-38,-3,-68
};
static ap_int<EIGHTBITS> fc_weight_opt[672] = {
		9,7,52,37,37,18,-23,46,38,-9,16,-1,17,-18,22,-10,11,6,59,38,0,20,-15,55,30,-9,11,-8,-3,-5,19,9,1,17,54,24,-18,19,-14,49,-18,-10,-19,-1,-23,-3,-13,11,-5,26,46,-11,-21,19,-5,49,-20,-4,-17,-24,-16,-3,-9,23,17,24,26,11,-11,9,-8,28,-4,-4,-13,-8,-17,5,-2,17,15,22,14,3,-22,15,-5,33,-29,2,-17,-2,-20,20,-7,25,12,27,8,-13,-21,15,15,8,-24,5,-23,-23,-40,18,-3,35,14,33,18,-31,-43,36,31,-6,-31,10,-23,-25,-34,36,-23,31,-4,27,14,-35,-29,50,44,-2,-21,1,-26,-20,-11,44,-21,28,-12,10,20,-25,-11,31,41,9,-10,-23,-12,-20,-2,37,-17,13,-15,8,9,-23,-11,20,20,8,-8,-31,-21,-21,-2,22,-11,9,-9,11,3,-27,-3,-10,14,-10,-5,-36,-16,-21,-10,22,-12,9,-14,9,-1,-29,-2,-14,20,-10,-5,-28,-10,-11,-8,20,-5,6,-10,6,-1,-23,-2,-8,6,-8,-2,-16,-2,-12,-23,14,-6,9,-6,2,-1,-34,-10,-5,8,-6,-10,-12,-9,-25,-23,13,-3,5,-2,0,2,-26,-2,-2,8,0,2,-13,3,-23,-21,8,2,8,0,10,-1,-14,18,-5,6,-3,5,-19,3,-18,-23,9,10,-1,-4,-3,3,-18,5,-5,2,1,8,-33,5,-24,-19,5,10,-8,-10,-5,3,-14,0,-6,2,1,3,-22,-10,-22,-16,6,3,-10,-4,-7,1,-16,15,-1,-2,-2,12,-13,9,-8,-10,4,18,4,-2,4,4,-2,8,-1,-5,-5,6,-5,3,-7,-11,-4,12,-3,-15,-13,-55,-32,-37,-20,25,-56,-36,9,-12,-7,-13,19,-22,9,-10,-16,-52,-36,-6,-24,7,-57,-32,9,-8,-1,-1,8,-26,-1,-2,-13,-54,-22,26,-16,15,-53,20,10,24,12,20,0,7,-8,-1,-23,-51,1,20,-18,4,-50,19,4,16,15,19,-1,11,-20,-23,-24,-31,-11,10,-12,4,-25,5,5,21,4,13,-6,1,-21,-14,-20,-16,-11,19,-14,0,-25,23,-3,16,8,23,-16,5,-26,-14,-30,-11,13,23,-12,-18,-11,26,-6,22,16,33,-20,1,-35,-13,-33,-19,31,42,-36,-33,8,31,-11,20,25,36,-36,22,-34,3,-28,-15,33,27,-53,-45,3,21,-2,26,22,11,-45,21,-29,10,-11,-21,21,10,-33,-43,-11,9,21,11,20,1,-38,15,-14,14,-8,-10,23,10,-20,-22,-9,6,31,21,20,1,-24,9,-10,8,-12,-4,25,1,9,-16,9,4,35,16,21,9,-24,11,-10,12,-12,0,29,1,13,-17,9,4,27,10,11,6,-22,4,-8,10,-7,0,21,1,8,-9,6,1,12,3,11,20,-14,5,-9,5,-3,0,35,8,3,-13,5,9,12,9,23,23,-12,3,-8,-2,-3,-4,26,3,1,-11,-1,-3,9,-5,22,21,-10,-2,-5,-4,-2,0,16,-18,4,-6,2,-7,18,-5,17,22,-10,-14,-9,7,1,-3,18,-5,4,-4,-2,-10,33,0,20,15,-6,-13,10,10,6,-4,13,2,5,-5,-2,-3,11,5,20,15,-7,-1,15,1,2,-2,14,-15,0,4,1,-13,12,-13,8,6,-4,-17,-7,6,-4,-5,2,-10,1,4,3,-6,12,-7,12,10,2,-15,-6
};
static ap_int<EIGHTBITS> fc_bias_opt[2] = {
		-8,12
};
ap_int<SIXTEENBITS> sigmoid_opt(ap_int<SIXTEENBITS> x)
{
	ap_int<SIXTEENBITS> result;
	ap_int<SIXTEENBITS> abs_x = abs(x);
	ap_int<SIXTEENBITS> denomonator = 1 + abs_x;
	ap_int<TWFOURBITS> tmp = (x * 128);
	result = tmp / denomonator;
	return(result);
}
ap_int<SIXTEENBITS> max_opt(ap_int<SIXTEENBITS> a, ap_int<SIXTEENBITS> b)
{
	if(a >= b)
	{
		return(a);
	}
	else // a < b
	{
		return(b);
	}
}
int mod_opt(int x, int y)
{
	  int r = x % y;
	  if (r < 0) {
	    r += y;
	  }
	  return (r);
}
void input_src(hls::stream<ap_int<EIGHTBITS>> &data, hls::stream<ap_int<EIGHTBITS>> & packed_src) {
	int read_index = 0;
	ap_int<EIGHTBITS> tmp;
	input_loop:while(read_index < 1600)
	{
#pragma HLS PIPELINE II=1
		tmp = data.read();
		packed_src.write(tmp);
		read_index = read_index + 1;
	}
	read_index=0;
}// it is a function to pack the data_input
void DepthwiseConvulution_opt(hls::stream<ap_int<EIGHTBITS>> &input, hls::stream<ap_int<SIXTEENBITS>> &output)
{
	int i=0, j=0;
	int i_index;
	int j_index;
	int index = 0;
	int count = 0;
	int index_slice=0;
	ap_int<EIGHTBITS> onesample_tmp[4],depthwisekernel_opt_tmp[4],pointwisekernel_opt_tmp[4];
	ap_int<SIXTEENBITS> data_separate_conv1_1[199*16];
	ap_int<EIGHTBITS> datatmp[1600];
	ap_int<TWFOURBITS> data_separate_conv1_1_tmp[4*199];
#pragma HLS ARRAY_PARTITION dim=1 factor=4 type=cyclic variable=data_separate_conv1_1_tmp
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=depthwisekernel_opt
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=pointwisekernel_opt
#pragma HLS ARRAY_PARTITION dim=1 factor=4 type=cyclic variable=datatmp

	ap_int<TWFOURBITS> conv_tmp[4], tmp;
	dscinput_loop1:for(i=0;i<1600;i++)
	{
		datatmp[i] = input.read();
	}
	/*DepthWise Kernel*/
	DepthwiseKernelLoop:
	for(i=0;i<796;i++)
	{
#pragma HLS PIPELINE
		i_index = floor((i)/4)*2;
		j_index = mod_opt(i,4);
		index = i_index + j_index*400;
		ap_int<EIGHTBITS> datatmppoint[4];
		ap_int<EIGHTBITS> depthwisekerneltmp[4];
		count = 0;
		depth_1_loop:while(count<4)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=4
//#pragma HLS UNROLL
			datatmppoint[count] = datatmp[index +count];
			count++;
		}
		count = 0;
		depth_2_loop:while(count<4)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=4
#pragma HLS UNROLL
			depthwisekerneltmp[count] = depthwisekernel_opt[j_index*4 + count];
			count++;
		}
		count=0;
		tmp = 0;
		depth_3_loop:for(j=0;j<4;j++)
		{
#pragma HLS UNROLL
			tmp = tmp + depthwisekerneltmp[j]*datatmppoint[j];
		}
		data_separate_conv1_1_tmp[i] = tmp;
	}
	/* PointWise Kernel */
	PointWiseKernel:
	for(i=0;i<3184;i++)
	{
#pragma HLS PIPELINE
		i_index = floor(i/16)*4;
		j_index = mod_opt(i,16);
		PWK_1_loop:while(count<4)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=4
#pragma HLS UNROLL
			conv_tmp[count] = data_separate_conv1_1_tmp[i_index+count];
			count++;
		}
		count = 0;
		PWK_2_loop:while(count<4)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=4
#pragma HLS UNROLL
			pointwisekernel_opt_tmp[count] = pointwisekernel_opt[j_index*4+count];
			count++;
		}
		count = 0;
		tmp = 0;
		PWK_3_loop:for(j=0;j<4;j++)
		{
#pragma HLS UNROLL
			tmp = tmp + conv_tmp[j]*pointwisekernel_opt_tmp[j];
		}
		tmp = tmp >>7;
		data_separate_conv1_1[i] = tmp;
		data_separate_conv1_1[i] = data_separate_conv1_1[i] + SeparateBias_opt[j_index];

		tmp = max_opt(0, data_separate_conv1_1[i]);
		output.write(tmp);
	}
}
void BatchNormAndMaxP_opt(hls::stream<ap_int<SIXTEENBITS>> &input, hls::stream<ap_int<SIXTEENBITS>> &output)
{
// input is data_separate_conv1_1_relu
// output is data_maxpooling_1
	int i_index;
	int j_index;
	int i=0;
	int j=0;
	ap_int<EIGHTBITS>  fenmu_opt=0;
	ap_int<SIXTEENBITS> data_batch_Normalization_1[199*16], data_BN_MAXP[198*16];
	ap_int<SIXTEENBITS> TmpMaxData=0;
	ap_int<SIXTEENBITS> data_separate_conv1_1_relu[3184], data_maxpooling_1[1056];

	ap_int<SIXTEENBITS> x_xbar=0, Fenzi=0;
	for(i=0;i<3184;i++)
	{
		data_separate_conv1_1_relu[i] = input.read();
	}
	/*BatchNormlization Layer*/
	BNLoop:
	for(i=0;i<3184;i+=16)
	{
		i_index = floor(i/16)*16;
		j_index = mod_opt(i,16);
		bnloop_1:
		for(j=0;j<16;j++)
		{
#pragma HLS PIPELINE
			fenmu_opt = Fenmu_opt[j_index+j];
			x_xbar = data_separate_conv1_1_relu[i+j] - BatchN_tMean_opt[j_index+j];
			Fenzi = BatchN_Scale_opt[j_index+j]*x_xbar;
			data_batch_Normalization_1[i+j] = Fenzi/fenmu_opt + BatchN_Offset_opt[j_index+j];
		}
	}
	/* mid processing, delete the last few data*/
	midprocessing_bn_loop:
	for(i=0;i<3168;i++)
	{
		data_BN_MAXP[i] = data_batch_Normalization_1[i];
	}
	/* maxpooling_1 */
	int count_maxpool1 = 0;
	int index1,index2,index3;
	loop5:for ( i=0;i<3137;i=i+48)
	{
		i_index = floor(i/48)*48;
		loop5_1:for( j =0;j<16;j++)
		{
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=data_BN_MAXP type=cyclic factor=16
			index1 = i_index+j;
			index2 = i_index+j+16;
			index3 = i_index+j+32;
			TmpMaxData = max_opt(data_BN_MAXP[index1], data_BN_MAXP[index2]);
			TmpMaxData = max_opt(TmpMaxData, data_BN_MAXP[index3]);
			data_maxpooling_1[count_maxpool1] = TmpMaxData;

			output.write(data_maxpooling_1[count_maxpool1]);
			count_maxpool1++;
		}
	}
}
void Conv2AndMaxP_opt(hls::stream<ap_int<SIXTEENBITS>> &input, hls::stream<ap_int<SIXTEENBITS>> &output)
{
	// input  is data_maxpooling_1[66*16]
	// output is data_maxpooling_2[336]
	int i_index;
	int j_index;
	int i=0;
	int j=0;
	int index;
	ap_int<EIGHTBITS> weights_tmp[64];
	ap_int<SIXTEENBITS> conv_tmp_2[64], data_conv1d_2[63*16], data_conv1d_2_relu[63*16];
	ap_int<SIXTEENBITS> TmpMaxData;
	ap_int<SIXTEENBITS> data_maxpooling_1[66*16], data_maxpooling_2[336];
	ap_int<SIXTEENBITS> tmp1, tmp2, tmp3;
#pragma HLS ARRAY_PARTITION variable=Conv2_Weights_opt type=cyclic factor=64
#pragma HLS ARRAY_PARTITION variable=data_maxpooling_1 type=cyclic factor=16
	int count = 0;
	/* pack to input */
	pack2data_loop:
	for(i=0;i<66*16;i++)
	{
		data_maxpooling_1[i] = input.read();
	}
	/* Convolution2 */
	Conv2_loop:
	for(i=0;i<1008;i++)
	{
#pragma HLS PIPELINE
		i_index = floor(i/16)*16;
		j_index = mod_opt(i,16);
		count=0;
		Conv2_1_loop:
		while(count<64)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
#pragma HLS UNROLL FACTOR=4
			conv_tmp_2[count] = data_maxpooling_1[i_index+count];
			count++;
		}
		count = 0;
		Conv2_2_loop:
		while(count<64)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=64
#pragma HLS UNROLL
			weights_tmp[count] = Conv2_Weights_opt[j_index*64+count];
			count++;
		}
		count=0;
		ap_int<TWFOURBITS> tmp = 0;
		Conv2_3_loop:for(j=0;j<64;j++)
		{
			tmp = tmp + weights_tmp[j]*conv_tmp_2[j];
		}
		data_conv1d_2[i] = tmp >> 7;
		tmp=0;
		data_conv1d_2[i] = data_conv1d_2[i] + Conv2_Bias_opt[j_index];
	}
	/* ReLU */
	Conv2_relu_loop:for(i=0;i<1008;i++)
	{
		data_conv1d_2_relu[i] = max_opt(0, data_conv1d_2[i]);
	}
	/* maxpooling 2 */
	int count_maxpool2 = 0;
	Conv2_maxp_loop:for(i=0;i<1008;i+=48)
	{
		i_index = floor(i/48)*48;
		loop8_1:for( j=0;j<16;j++)
		{
#pragma HLS PIPELINE
#pragma HLS ARRAY_RESHAPE variable= data_conv1d_2_relu dim=1 factor=16 type=cyclic
			tmp1 = data_conv1d_2_relu[i_index+j];
			tmp2 = data_conv1d_2_relu[i_index+j+16];
			tmp3 = data_conv1d_2_relu[i_index+j+32];

			TmpMaxData = max_opt(tmp1, tmp2);
			TmpMaxData = max_opt(TmpMaxData, tmp3);
			data_maxpooling_2[count_maxpool2] = TmpMaxData;

			output.write(data_maxpooling_2[count_maxpool2]);
			count_maxpool2++;
		}
	}
	count_maxpool2=0;

}
void DenseAndSigmoid_opt(hls::stream<ap_int<SIXTEENBITS>> &input,  hls::stream<ap_int<SIXTEENBITS>> &output)
{
	// input is data_maxpooling_2[336];
	// output is  data_dense_sigmoid[2];
	int i_index;
	int j_index;
	int i=0;
	int j=0;
	int index;
	ap_int<TWFIVEBITS> tmp = 0;
	ap_int<EIGHTBITS> fc_weitht_tmp[336];
	ap_int<SIXTEENBITS> data_dense[2], data_maxpooling_2_tmp[336], data_dense_sigmoid_tmp[2], data_maxpooling_2[336];
	ap_int<SIXTEENBITS> data_dense_sigmoid[2];
	ap_int<SIXTEENBITS> sigmoid_tmp;
#pragma HLS ARRAY_PARTITION variable=fc_weight_opt type=block factor=2

	/* pack to data array */
	densesigmoid_pack2data:for(i=0;i<336;i++)
	{
#pragma HLS PIPELINE II=1
		data_maxpooling_2[i] = input.read();
	}
	/* Dense Layer */
	dense_loop:
	for(i=0;i<2;i++)
	{
		int count =0;
		denseloop1:while(count<336)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=336
#pragma HLS PIPELINE
			fc_weitht_tmp[count] = fc_weight_opt[i*336+count];
			count++;
		}
		count = 0;
		denseloop2:for(j = 0;j<336;j++)
		{
			tmp = tmp + fc_weitht_tmp[j]*data_maxpooling_2[j];
		}
		data_dense[i] = tmp >> 7;
		data_dense[i] = data_dense[i] + fc_bias_opt[i];
		tmp = 0;
	}
	/* Sigmoid Activation Function*/
	sigmoid_loop:for( i=0;i<2;i++)
	{
		sigmoid_tmp = sigmoid_opt(data_dense[i]);
		 data_dense_sigmoid_tmp[i] = sigmoid_tmp;
	}
	if (data_dense_sigmoid_tmp[1]>=data_dense_sigmoid_tmp[0])
	{
		data_dense_sigmoid[0]=0;
		data_dense_sigmoid[1]=1;
	}
	else
	{
		data_dense_sigmoid[0]=1;
		data_dense_sigmoid[1]=0;
	}

	output.write(data_dense_sigmoid[1]);

}
void dwcnnopt(hls::stream<ap_int<EIGHTBITS>> &datainput, hls::stream<ap_int<SIXTEENBITS>> &out)
//ap_int<SIXTEENBITS> dwcnnopt(ap_int<EIGHTBITS> datainput[1600])
{
#pragma HLS INTERFACE mode=axis register_mode=both port=out register
#pragma HLS INTERFACE mode=axis register_mode=both port=datainput register
#pragma HLS INTERFACE mode=s_axilite bundle=CTRL port=return
#pragma HLS DATAFLOW

	hls::stream<ap_int<EIGHTBITS>> data; 					// shape is 1600
	hls::stream<ap_int<SIXTEENBITS>> data_separate_conv1_1_relu;// shape is 3184
	hls::stream<ap_int<SIXTEENBITS>> data_maxpooling_1; 		// shape is 66*16
	hls::stream<ap_int<SIXTEENBITS>> data_maxpooling_2;			// shape is 336
	hls::stream<ap_int<SIXTEENBITS>> result;

#pragma HLS STREAM variable=data depth=100
#pragma HLS STREAM variable=data_separate_conv1_1_relu depth=100
#pragma HLS STREAM variable=data_maxpooling_1 depth=100
#pragma HLS STREAM variable=data_maxpooling_2 depth=100

	input_src(datainput, data);
	DepthwiseConvulution_opt(data, data_separate_conv1_1_relu);
	BatchNormAndMaxP_opt(data_separate_conv1_1_relu, data_maxpooling_1);
	Conv2AndMaxP_opt(data_maxpooling_1, data_maxpooling_2);
	DenseAndSigmoid_opt(data_maxpooling_2, out);

}
