#ifndef INCLUDE_DWCNN_QUANTIZED_OPT_H_
#define INCLUDE_DWCNN_QUANTIZED_OPT_H_
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "ap_fixed.h"	//ap_fixed<18,6,AP_TRN_ZERO,AP_SAT>		<W,I,Q,O,N>
#include "ap_int.h"	//ap_int<N> or ap_uint<N>, 1<=N<=1024
#include "hls_stream.h"
const int TWFIVEBITS = 25; // for debug
const int TWFOURBITS = 24; // maximum intermediate, accumulation, 24-bits
const int SIXTEENBITS = 16; 
const int EIGHTBITS = 8;
const int SLICE_LEGTH = 1600; // input size, line format

void dwcnnopt(hls::stream<ap_int<EIGHTBITS>> &datainput, hls::stream<ap_int<SIXTEENBITS>> &out);

#endif
