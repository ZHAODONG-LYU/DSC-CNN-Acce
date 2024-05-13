#include <stdint.h>
#include <stdlib.h>
#include "dwcnn_quantized_opt.h"
#include "math.h"






int sigmoid_opt(int x)
{
	int abs_x = abs(x);
	int denomonator = 1 + abs_x;
	int result = (x * 128) / denomonator;
	return (result);
}

int max_opt(int a, int b)
{
	if (a>=b)
	{
		return (a);
	}
	else
	{
		return (b);
	}
}

void slice1600_4_opt(int8_t a[1600], int i_start, int8_t out[4])
{
	int counts = 0;
	while(counts<4)
	{
#pragma HLS UNROLL
		out[counts] = a[counts+i_start];
		counts++;
	}
}
 void slice16_4_opt( const int8_t a[16], int i_start, int8_t out[4])
{
	int counts = 0;
	while(counts<4)
	{
#pragma HLS UNROLL
		out[counts] = a[counts+i_start];
		counts++;
	}

}
 void slice796_4_opt(int16_t a[796], int i_start, int16_t out[4])
{
	int counts = 0;
	while(counts<4)
	{
#pragma HLS UNROLL
		out[counts] = a[counts+i_start];
		counts++;
	}

}
 void slice64_4_opt( const int8_t a[64], int i_start, int8_t out[4])
{
	int counts = 0;
	while(counts<4)
	{
#pragma HLS UNROLL
		out[counts] = a[counts+i_start];
		counts++;
	}
}
 void slice3184_3168_opt(int16_t a[3184], int i_start,  int16_t out[3168])
{
	int counts = 0;
	while(counts<3168)
	{
#pragma HLS PIPELINE
		out[counts] = a[counts+i_start];
		counts++;
	}
}
 void slice1056_64_opt(int16_t a[1056], int i_start, int16_t out[64])
{
	int counts = 0;
	while(counts<64)
	{
#pragma HLS UNROLL
		out[counts] = a[counts+i_start];
		counts++;
	}
}
 void slice1024_64_opt( const int8_t a[1024], int i_start, int8_t out[64])
{
	int counts = 0;
	while(counts<64)
	{
#pragma HLS UNROLL
		out[counts] = a[counts+i_start];
		counts++;
	}
}
void slice672_336_opt(const  int8_t a[672], int i_start, int8_t out[336])
{
	int counts = 0;
	while(counts<336)
	{
		out[counts] = a[counts+i_start];
		counts++;
	}
}
void slice336_336_opt(int16_t a[336], int i_start, int i_end, int16_t out[336])
{
	int counts = 0;
	while(counts<336)
	{
		out[counts] = a[counts+i_start];
		counts++;
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
void DepthwiseConvulution_opt(int8_t data_input[1600],  const int8_t depthwisekernel_opt[16],  const int8_t pointwisekernel_opt[64], int16_t data_separate_conv1_1_relu[199*16])
{
	int i=0, j=0;
	int i_index;
	int j_index;
	int index, tmp = 0;
	int8_t onesample_tmp[4],depthwisekernel_opt_tmp[4],pointwisekernel_opt_tmp[4];
	int16_t data_separate_conv1_1[199*16], data_separate_conv1_1_tmp[199*4],conv_tmp[4];



	/* DSC layer */
	loop1:for( i = 0;i < 796;i++)
	{
#pragma HLS PIPELINE
		i_index = floor((i)/4)*2;
		j_index = mod_opt(i,4);
		index = i_index + j_index*400;
		slice1600_4_opt( data_input, index, onesample_tmp);
		slice16_4_opt( depthwisekernel_opt, j_index*4, depthwisekernel_opt_tmp);
		int tmp = 0;
		for(j = 0; j<4;j++)
		{
#pragma HLS UNROLL
			tmp = tmp + onesample_tmp[j]*depthwisekernel_opt_tmp[j];
		}
		data_separate_conv1_1_tmp[i] = tmp;
	}
	loop2:for ( i = 0;i<3184;i++)
	{
#pragma HLS PIPELINE
		i_index = floor(i/16)*4;
		j_index = mod_opt(i,16);
		slice796_4_opt( data_separate_conv1_1_tmp,i_index, conv_tmp);
		slice64_4_opt( pointwisekernel_opt,  j_index*4, pointwisekernel_opt_tmp);
		int tmp=0;
		for (j=0; j<4;j++)
		{
#pragma HLS UNROLL
			tmp = tmp + conv_tmp[j]*pointwisekernel_opt_tmp[j];
		}
		tmp = tmp >> 7;
		data_separate_conv1_1[i] = tmp;
		data_separate_conv1_1[i] = data_separate_conv1_1[i] + SeparateBias_opt[j_index];
	}
	loop3:for ( i =0;i<3184;i++)/* ReLU */
	{
#pragma HLS PIPELINE
		data_separate_conv1_1_relu[i] = max_opt(0, data_separate_conv1_1[i]);
	}

}
void BatchNormAndMaxP_opt(int16_t data_separate_conv1_1_relu[3184],  const int8_t BatchN_Scale_opt[16],  const int8_t BatchN_tMean_opt[16],  const int8_t BatchN_Offset_opt[16],  const int8_t Fenmu_opt[16], int16_t data_maxpooling_1[66*16])
{

	int i_index;
	int j_index;
	int i=0;
	int j=0;
	int8_t  fenmu_opt=0;
	int16_t data_batch_Normalization_1[199*16], data_BN_MAXP[198*16];
	int16_t TmpMaxData=0;
#pragma HLS ARRAY_PARTITION variable=data_BN_MAXP cyclic factor=16 dim=1

	int16_t x_xbar=0, Fenzi=0;
	/* batchNormlization */
		loop4:for ( i=0; i<3184;i+=16)
		{
			i_index = floor(i/16)*16;
			j_index = mod_opt(i,16);
			loop4_1:for (j = 0;j<16;j++)
			{
#pragma HLS UNROLL
				fenmu_opt = Fenmu_opt[j_index+j];
				x_xbar = data_separate_conv1_1_relu[i+j] - BatchN_tMean_opt[j_index+j];
				Fenzi = BatchN_Scale_opt[j_index+j]*x_xbar;
				data_batch_Normalization_1[i+j] = Fenzi/fenmu_opt + BatchN_Offset_opt[j_index+j];
			}
		}
		/* mid processing */
		slice3184_3168_opt(data_batch_Normalization_1, 0, data_BN_MAXP);
		/* maxpooling1 */
		int count_maxpool1 = 0;
		int index1,index2,index3;
		loop5:for ( i=0;i<3137;i=i+48)
		{
			i_index = floor(i/48)*48;
			loop5_1:for( j =0;j<16;j++)
			{
				index1 = i_index+j;
				index2 = i_index+j+16;
				index3 = i_index+j+32;
				TmpMaxData = max_opt(data_BN_MAXP[index1], data_BN_MAXP[index2] );
				TmpMaxData = max_opt(TmpMaxData, data_BN_MAXP[index3]);
				data_maxpooling_1[count_maxpool1] = TmpMaxData;
				count_maxpool1++;
			}
		}
}
void Conv2AndMaxP_opt(int16_t data_maxpooling_1[66*16], const int8_t Conv2_Weights_opt[1024],  const int8_t Conv2_Bias_opt[16], int16_t data_maxpooling_2[336])
{
	int i_index;
	int j_index;
	int i=0;
	int j=0;
	int index, tmp = 0;
	int8_t weights_tmp[64];
	int16_t conv_tmp_2[64], data_conv1d_2[63*16], data_conv1d_2_relu[63*16];
	int16_t TmpMaxData=0;


	/* conv1d_2 */
	loop6:for ( i=0;i<1008;i++)
	{
		i_index = floor(i/16)*16;
		j_index = mod_opt(i,16);
		slice1056_64_opt( data_maxpooling_1, i_index, conv_tmp_2);
		slice1024_64_opt( Conv2_Weights_opt, j_index*64, weights_tmp);
		int tmp = 0;
		for(j = 0; j<64 ; j++)
		{
#pragma HLS UNROLL
			tmp = tmp + conv_tmp_2[j]*weights_tmp[j];
		}
		data_conv1d_2[i] = tmp >> 7;
		data_conv1d_2[i] = data_conv1d_2[i] + Conv2_Bias_opt[j_index];
	}
	/* ReLU */
	loop7:for ( i=0;i<1008;i++)
	{
#pragma HLS PIPELINE
		data_conv1d_2_relu[i] = max_opt(0, data_conv1d_2[i]);
	}

	/* maxpooling2  */
	int count_maxpool2 = 0;
	loop8:for ( i=0;i<1008;i=i+48)
	{
		i_index = floor(i/48)*48;
		loop8_1:for( j=0;j<16;j++)
		{
			TmpMaxData = max_opt(data_conv1d_2_relu[i_index+j], data_conv1d_2_relu[i_index+j+16]);
			TmpMaxData = max_opt(data_conv1d_2_relu[i_index+j+32], TmpMaxData);
			data_maxpooling_2[count_maxpool2] = TmpMaxData;
			count_maxpool2++;
		}
	}
}
void DenseAndSigmoid_opt(int16_t data_maxpooling_2[336],  const int8_t fc_weight_opt[672],  const int8_t fc_bias_opt[2],  int16_t data_dense_sigmoid[2])
{
	int i_index;
	int j_index;
	int i=0;
	int j=0;
	int index, tmp = 0;
	int8_t fc_weitht_tmp[336];
	int16_t data_dense[2], data_maxpooling_2_tmp[336];
	int16_t data_dense_sigmoid_tmp[2];
	/* dense layer */
	loop9:for ( i=0;i<2;i++)
	{
		slice672_336_opt( fc_weight_opt, i*336, fc_weitht_tmp);
		int tmp = 0;
		for(j = 0; j< 336; j++)
		{
#pragma HLS UNROLL FACTOR=16
			tmp = tmp + fc_weitht_tmp[j]*data_maxpooling_2[j];
		}
		data_dense[i] = tmp >> 7;
		data_dense[i] = data_dense[i] + fc_bias_opt[i];
	}
	/* sigmoid_opt */
	loop10:for( i=0;i<2;i++)
	{
		data_dense_sigmoid_tmp[i] = sigmoid_opt(data_dense[i]);
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
}
int dwcnnopt(int8_t data_input[1600])
{
#pragma HLS DATAFLOW
	int16_t data_separate_conv1_1_relu[199*16];
	int16_t data_maxpooling_1[66*16];
	int16_t data_maxpooling_2[21*16];
	int16_t data_dense_sigmoid[2];
	int     predict_result;
#pragma HLS ARRAY_PARTITION variable=Conv2_Weights_opt block factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=depthwisekernel_opt block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=pointwisekernel_opt cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=fc_weight_opt block factor=2 dim=1


	DepthwiseConvulution_opt(data_input, depthwisekernel_opt, pointwisekernel_opt, data_separate_conv1_1_relu);
	BatchNormAndMaxP_opt(data_separate_conv1_1_relu, BatchN_Scale_opt, BatchN_tMean_opt, BatchN_Offset_opt, Fenmu_opt, data_maxpooling_1);
	Conv2AndMaxP_opt(data_maxpooling_1, Conv2_Weights_opt, Conv2_Bias_opt, data_maxpooling_2);
	DenseAndSigmoid_opt(data_maxpooling_2, fc_weight_opt, fc_bias_opt, data_dense_sigmoid);


	predict_result = data_dense_sigmoid[1];
	return (predict_result);

}
