#include "FX.cu"
#include "config.h"
#include "utils.cuh"

__global__ void winogradConvolution(float *intput, int batch, int channel,
                                    int size, int k, float *workspace,
                                    float *output, int tile_dim);

__device__ __forceinline__ void prefetch_filter_tile(float *filter, float *tile,
                                                     int k);

__device__ __forceinline__ void prefetch_input_tile(float *input, float *tile,
                                                    int batch, int size,
                                                    int tile_dim, short mask);
__device__ __forceinline__ void
load_and_transform_input_tile(float *input_tile, float *input_smem, int batch,
                              int channel, int size);

cudaError_t convolutionFwd(float *input, int batch, int channel, int size,
                           float *filter, int k, int ksize, float *output,
                           float *workspace, int tile_size, int tile_dim) {

    int tile_2d = tile_size * tile_size; // 4*4
    int tile_2d_dim = tile_dim * tile_dim;
    size_t smem_size = (16 * BC * (BN + BK)) << 2;
    FX<<<dim3(k / BK, channel / BC), dim3(BK, BC)>>>(filter, workspace, k,
                                                     channel, ksize);
    winogradConvolution<<<dim3(batch / BN, tile_2d_dim, k / BK), dim3(BN, BC),
                          smem_size>>>(input, batch, channel, size, k,
                                       workspace, output, tile_dim);
    return cudaGetLastError();
}

__global__ void winogradConvolution(float *input, int batch, int channel,
                                    int size, int k, float *filter,
                                    float *output, int tile_dim) {
    extern __shared__ float smem[];
    float *input_smem = smem;
    float *filter_smem = smem + (16 * BC * BN);
    short mask = 0xffff;
    if ((blockIdx.y / tile_dim) == 0) {
        mask &= 0xfff0;
    }
    if ((blockIdx.y / tile_dim) == tile_dim - 1) {
        mask &= (!(size & 2)) ? 0x0fff : 0x00ff;
    }
    if (!((blockIdx.y + 1) % tile_dim)) {
        mask &= (!(size % 2)) ? 0x7777 : 0x3333;
    }
    if (!((blockIdx.y) % tile_dim)) {
        mask &= 0xeeee;
    }
    float input_tile[16];  // 4*4 input tile;
    float filter_tile[32]; // 2*4*4 filter tile;

    // double buff
    float4 input_frag_mem[8];  // 4*4 float in each buffer
    float4 filter_frag_mem[8]; // 4*4 float in each buffer

    float4 *input_frag = input_frag_mem, *input_frag_buff = input_frag_mem + 4;
    float4 *filter_frag = filter_frag_mem,
           *filter_frag_buff = filter_frag_mem + 4;

    float4 *load_input = nullptr, *load_filter = nullptr, *swap;

    float4 accumulator[2][16] = {0.0f}; // local result

    int input_frag_offset = 2 * BC * BN;
    int filter_frag_offset = 2 * BC * BK;

    prefetch_input_tile(input, input_tile, batch, size, tile_dim, mask);
    prefetch_filter_tile(filter, filter_tile, k);
    for (int iter = 0; iter < channel; iter += BC) {
        load_input = (float4 *)(input_smem + threadIdx.y * BC * BN);
        load_filter = (float4 *)(filter_smem + threadIdx.y * BC * BK);

        load_and_transform_input_tile();
    }
}

__device__ __forceinline__ void prefetch_filter_tile(float *filter, float *tile,
                                                     int k) {
    int tile_idx = blockIdx.z * BK + (threadIdx.y * k << 4) + threadIdx.x;
    int offset;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        offset = (i * k << 2);
#pragma unroll;
        for (int j = 0; j < 4; j++) {
            tile[(i << 2) + j] = filter[tile_idx + offset + j * k];
            tile[(i << 2) + j + 16] = filter[tile_idx + offset + j * k + BN];
        }
    }
}

__device__ __forceinline__ void prefetch_input_tile(float *input, float *tile,
                                                    int batch, int size,
                                                    int tile_dim, short mask) {
    int tile_idx = (blockIdx.y % tile_dim) * batch * 2 +
                   (blockIdx.y / tile_dim) * batch * size * 2 +
                   blockIdx.x * BN + (threadIdx.y - 1) * (batch * size * size) +
                   (threadIdx.x % batch);
    int offset, x;
    if (mask == 0xffff) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            offset = i * batch * size;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                x = (i << 2) + j;
                tile[x] = input[tile_idx + offset + j * batch];
            }
        }
    } else {
        for (int i = 0; i < 4; i++) {
            offset = i * batch * size;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                x = (i << 2) + j;
                tile[x] = 0;
                if (mask & (1 << x)) {
                    tile[x] = input[tile_idx + offset + j * batch];
                }
            }
        }
    }
}
__device__ __forceinline__ void
load_and_transform_input_tile(float *input_tile, float *input_smem, int batch,
                              int channel, int size) {
    // *
    // *     BT matrix                 input tile             B matrix
    // *  |   1,   0,  -1,   0|   |  x1,  x2,  x3,  x4|    |   1,   0,   0,  0|
    // *  |   0,   1,   1,   0|   |  x5,  x6,  x7,  x8|    |   0,   1,  -1,  1|
    // *  |   0,  -1,   1,   0|   |  x9, x10, x11, x12|    |  -1,   1,   1,  0|
    // *  |   0,   1,   0,  -1|   | x13, x14, x15, x16|    |   0,   0,   0, -1|
    // *
#define visit(tile, i, j) (tile[((i) << 2) + (j)])

    float buff[3];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        buff[0] = input_tile[j];
        buff[1] = input_tile[j + 4];
        buff[3] = input_tile[j + 8];

        input_tile[j] = buff[0] - buff[2];
        input_tile[j + 4] = buff[1] + buff[2];
        input_tile[j + 8] = buff[2] - buff[1];
        input_tile[j + 12] = buff[1] - input_tile[j + 12];
    }
    int offset = BN * BC;
    int tile_idx = threadIdx.y * BN + threadIdx.x;
// * layout chwn -> 16 BC BN in shared memory
#pragma unroll
    for (int i = 0; i < 4; i++) {
        input_smem[tile_idx + ((i << 2)) * offset] =
            visit(input_tile, i, 0) - visit(input_tile, i, 2);
        input_smem[tile_idx + ((i << 2 + 1) * offset)] =
            visit(input_tile, i, 1) + visit(input_tile, i, 2);
        input_smem[tile_idx + ((i << 2) + 2) * offset] =
            visit(input_tile, i, 2) - visit(input_tile, i, 2);
        input_smem[tile_idx + ((i << 2) + 3) * offset] =
            visit(input_tile, i, 1) - visit(input_tile, i, 3);
    }
}