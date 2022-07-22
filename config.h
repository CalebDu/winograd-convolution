#pragma once

// #ifndef CONFIG_HEADER
// #define CONFIG_HEADER
#pragma once

#define BC 8
#define BK 64
#define BN 32

#define BN_P 40

// * default input and kernel size
#define BATCH 128
#define CHANNEL 256
#define SIZE 14
#define KFILTER 256
#define KSIZE 3

// * configuration of convolution
#define PAD_H 1
#define PAD_W 1
#define STR_H 1
#define STR_W 1
#define DIL_H 1
#define DIL_W 1

// #define M 2
#define ALPHA 4

__constant__ int access_w[2][32];
__constant__ int access_x[2][32];
__constant__ int access_out[2][16];
__constant__ int out_thread[4][4];

// access_f_s
const int aux[2][32] = {{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                         0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7},
                        {8,  8,  9,  9,  10, 10, 11, 11, 12, 12, 13,
                         13, 14, 14, 15, 15, 8,  8,  9,  9,  10, 10,
                         11, 11, 12, 12, 13, 13, 14, 14, 15, 15}};
// access_s
const int aux2[2][32] = {{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                          2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3},
                         {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
                          6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7}};
// access_out
const int aux3[2][16] = {
    {0, 1, 10, 11, 20, 21, 30, 31, 2, 3, 12, 13, 22, 23, 32, 33},
    {4, 5, 14, 15, 24, 25, 34, 35, 6, 7, 16, 17, 26, 27, 36, 37}};
const int aux4[4][4] = {
    {0, 4, 8, 12}, {32, 36, 40, 44}, {16, 20, 24, 28}, {48, 52, 56, 60}};

// #endif