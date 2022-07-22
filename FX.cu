
#include "config.h"
using gemm_function = float (*)(float *, int);

__device__ float row1(float *mat, int j) { return mat[j]; }
__device__ float row2(float *mat, int j) {
    return 0.5 * (mat[j] + mat[3 + j] + mat[6 + j]);
}
__device__ float row3(float *mat, int j) {
    return 0.5 * (mat[j] - mat[3 + j] + mat[6 + j]);
}

__device__ float row4(float *mat, int j) { return mat[j + 6]; }

__device__ float col1(float *mat, int j) { return mat[j]; }
__device__ float col2(float *mat, int j) {
    return 0.5 * (mat[j] + mat[j + 1] + mat[j + 2]);
}
__device__ float col3(float *mat, int j) {
    return 0.5 * (mat[j] - mat[j + 1] + mat[j + 2]);
}

__device__ float col4(float *mat, int j) { return mat[j + 2]; }

// *
// *     G matrix           filter              GT matrix
// *  |   1,   0,  0 |   |  x1,  x2,  x3|    |   1, 0.5, 0.5,  0|
// *  | 0.5, 0.5, 0.5|   |  x4,  x5,  x6|    |   0, 0.5,-0.5,  0|
// *  | 0.5,-0.5, 0.5|   |  x7,  x8,  x9|    |   0, 0.5, 0.5,  1|
// *  |   0,   0,   1|
// *
__global__ void FX(float *filter, float *workspace, int k, int channel,
                   int ksize) {
    auto tidx = threadIdx.x, tidy = threadIdx.y, bidx = blockIdx.x,
         bidy = blockIdx.y;
    int filter_global_offset = k * ksize * ksize;
    int filter_idx = tidx + tidy * filter_global_offset + bidx * BK +
                     bidy * BC * filter_global_offset;

    int workspace_global_offset = k * ALPHA * ALPHA;
    int workspace_idx = tidx + tidy * workspace_global_offset + bidx * BK +
                        bidy * BC * workspace_global_offset;

    float buffer[21]; // 4*3 + 3*3;
    float *local_filter = buffer, *local_Gxfilter = buffer + 9;

    for (int i = 0; i < 9; i++) {
        local_filter[i] = filter[filter_idx + i * k];
    }

    gemm_function row_fn[4] = {row1, row2, row3, row4};
    gemm_function col_fn[4] = {col1, col2, col3, col4};

    int offset, offset2;
    // G * filter = [4, 3] * [3, 3] = [4, 3]
    for (int i = 0; i < 4; i++) {
        offset = i * 3;
        for (int j = 0; j < 3; j++) {
            local_Gxfilter[j + offset] = (*row_fn[i])(local_filter, j);
        }
    }
    // Gfilter * GT = [4, 3] * [3, 4] = [4, 4]
    for (int i = 0; i < 4; i++) {
        offset = i * 3;
        offset2 = i << 2;
        for (int j = 0; j < 4; j++) {
            workspace[workspace_idx + (offset2 + j) * k] =
                (*col_fn[j])(local_Gxfilter, offset);
        }
    }
}