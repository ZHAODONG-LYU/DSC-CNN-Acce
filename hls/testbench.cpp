#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "dwcnn_quantized_opt.h"
#include <string.h>
#include "math.h"
#include "time.h"
#include <inttypes.h>
#include <iostream>
#include <fstream>


#define COLS 1600
#define MAX_LINE_LENGTH (COLS * 5)
#define SAMPLENUMS 21


using namespace std;


int main(void)
{
	// define some parameters and variations 
	ap_int<SIXTEENBITS> predict_result_opt[SAMPLENUMS];
	clock_t t_start, t_end;
    double duration;
    int i = 0, j = 0;
    hls::stream<ap_int<EIGHTBITS>> data_input;
    hls::stream<ap_int<SIXTEENBITS>> dataout;
    
    
    
    
    // initialize the prediction_rersult_opt
    for(i=0; i<SAMPLENUMS; i++)
    {
    	predict_result_opt[i] = 0;
    }
    i=0;
    ap_int<SIXTEENBITS> prediction_result_tmp=0;




    // open the file
	FILE * file = fopen("data_piece_120.csv","r");
    if (file == NULL) {
        perror("ERROR IN OPENING FILE DATA.CSV");
        return (1);
    }




	// start to process the stream
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file) != NULL) {
        /* to store int8_t data for a cols*/
        int8_t data[COLS];

        /* depart the string by ",", and transmute into int_8*/
        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < COLS) {
            data[col] = (int8_t)atoi(token);
            token = strtok(NULL, ",");
            col++;
        }
        mm2s:
        for(j=0;j<1600;j++)
        {
        	data_input.write((ap_int<EIGHTBITS>)data[j]);
        }
        /* here you can add some operations for the data array.*/
        
        dwcnnopt(data_input, dataout); // the main function
        
        s2mm:
        predict_result_opt[i] = dataout.read();
        std::cout<<"SAMPLE INDEX IS "<<i<<std::endl;
        i++;
    }




    /* close the file */
    fclose(file);
    i=0;
    FILE * filelabels = fopen("labels_piece_120.txt","r");
        if (filelabels == NULL) {
            perror("ERROR in OPENING FILE LABELS.TXT");
            return (1);
        }
     int Label[SAMPLENUMS];
     int arraySize = 0;
     int value;
     while (fscanf(filelabels, "%d", &value) == 1 && arraySize < SAMPLENUMS) {
           /* load int into array */
    	 Label[arraySize] = value;
         arraySize++;
       }
     fclose(filelabels);

     /* comparison and calculating the accuracy */
     int correctnums = 0;
     for (i=0;i<SAMPLENUMS;i++)
     {
    	 std::cout<<"LABEL IS "<<Label[i]<<" PREDICTION RESULT IS "<<predict_result_opt[i]<<std::endl;
    	if (Label[i] - predict_result_opt[i] == 0)
    	{
    		correctnums++;
    	}
    	else
    	{
    		std::cout<<"BAD SAMPLE INDEX IS "<<i<<std::endl;
    		return (1);
    	}
     }
     double accuracy = (double)((double)correctnums)/SAMPLENUMS *100;
     printf("CORRECT NUMS IS %d\n", correctnums);
     printf("ACCURACY IS %.2f%%\n", accuracy);
     if (correctnums > SAMPLENUMS-2)
     {
    	 printf("GOOD QUANTIZATION!\n");
     }
     else
     {
    	 printf("BAD  QUANTIZATION!\n");
     }
    return (0);
}
