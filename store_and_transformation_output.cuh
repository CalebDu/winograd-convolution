
#include "config.h"

// *
// *     AT matrix                 output tile           A matrix
// *  |   1,   1,   1,  0|   |  x1,  x2,  x3,  x4|     |   1,   0|
// *  |   0,   1,  -1, -1|   |  x5,  x6,  x7,  x8|     |   1,   1|
// *                         |  x9, x10, x11, x12|     |   1,  -1|
// *                         | x13, x14, x15, x16|     |   0,  -1|
// *

__device__ void transform_output_tile(float *result, float *output_tile,
                                      float *transform_tile, int out_size,
                                      int tile_dim, int round, int k,
                                      int out_thread[][4], short mask,
                                      int tile_idx, int output_global_offset) {
    for (int j = 0; j < 4; j++) {
        transform_tile[j] =
            output_tile[j] + output_tile[j + 4] + output_tile[j + 8];
        transform_tile[j + 4] =
            output_tile[j + 4] - output_tile[j + 8] + output_tile[j + 12];

        transform_tile[j + 8] = output_tile[j + 16] + output_tile[j + 4 + 16] +
                                output_tile[j + 8 + 16];
        transform_tile[j + 4 + 8] = output_tile[j + 4 + 16] -
                                    output_tile[j + 8 + 16] +
                                    output_tile[j + 12 + 16];
    }
    int idx = out_thread[round][threadIdx.x & 3] + (threadIdx.y >> 2);
    tile_idx += idx * output_global_offset;

    int offset1, offset2;
    for (int i = 0; i < 2; i++) {
        offset1 = i * 4;
        offset2 =
            i * (k * (tile_dim - (out_size & 1)) + (out_size & 1) * k / 2) * 2;
        if (mask & (1 << (i * 2))) {
            result[tile_idx + offset2] = transform_tile[offset1] +
                                         transform_tile[offset1 + 1] +
                                         transform_tile[offset1 + 2];
            result[tile_idx + offset2 + 2 * output_global_offset] =
                transform_tile[offset1 + 8] + transform_tile[offset1 + 1 + 8] +
                transform_tile[offset1 + 2 + 8];
        }

        if (mask & (1 << (i * 2 + 1))) {
            result[tile_idx + offset2 + k] = transform_tile[offset1 + 1] -
                                             transform_tile[offset1 + 2] -
                                             transform_tile[offset1 + 3];
            result[tile_idx + offset2 + 2 * output_global_offset + k] =
                transform_tile[offset1 + 1 + 8] -
                transform_tile[offset1 + 2 + 8] -
                transform_tile[offset1 + 3 + 8];
        }
    }
}
__device__ void store_output_tile(float4 accumulator[][16], float *shared_mem,
                                  float *result, int out_size, int tile_dim,
                                  int k, float4 *input_frag_mem,
                                  float4 *filter_frag_mem, int out_thread[][4],
                                  int access_s_out[][16]) {
    float4 *output_smem = (float4 *)shared_mem;
    float4 *accumulator_ptr = (float4 *)accumulator;
    float *output_tile = (float *)input_frag_mem;
    float *transform_tile = (float *)filter_frag_mem;

    short mask = 0xffff;
    if ((blockIdx.y / tile_dim == (tile_dim - 1)) && out_size % 2) {
        mask &= 0x0003;
    }
    if (!((blockIdx.y + 1) % tile_dim) && out_size % 2) {
        mask &= 0x00005;
    }

    int step = 0;
    int smem_idx = (threadIdx.y >> 2) * BN_P * 16 * 4 +
                   (threadIdx.y & 3) * BN_P + threadIdx.x;

    int output_global_offset = k * out_size * out_size;
    int tile_idx = blockIdx.z * k * out_size * out_size * BK +
                   (blockIdx.y & tile_dim) * k * 2 +
                   (blockIdx.y / tile_dim) * k * out_size * 2 +
                   blockIdx.x * BN + threadIdx.x;

    // 160 byte = 128 + 32 padding
    int offset = BN_P * 4;

    int offset1 = access_out[0][(threadIdx.x & 7) + ((threadIdx.x >> 4) >> 3)];
    int offset2 = access_out[1][(threadIdx.x & 7) + ((threadIdx.x >> 4) >> 3)];

    int offset3 = threadIdx.y * BN_P;
    int offset4 = BN << 4;

    float *output = (float *)output_smem;
    int idx = offset3, idx2 = idx + (BN_P * 8);

    for (int round = 0; round < 4; round++) {
        if (((!round || round == 1) && (threadIdx.x & 15) < 8) ||
            ((round == 2 || round == 3) && (threadIdx.x & 15) > 7)) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                output_smem[idx + i * offset4 + offset1] =
                    accumulator_ptr[step];
                output_smem[idx + i * offset4 + offset2] =
                    accumulator_ptr[step + 1];
                output_smem[idx + (i + 1) * offset4 + offset1] =
                    accumulator_ptr[step + 2];
                output_smem[idx + (i + 1) * offset4 + offset2] =
                    accumulator_ptr[step + 3];

                output_smem[idx2 + i * offset4 + offset1] =
                    accumulator_ptr[step + 16];
                output_smem[idx2 + i * offset4 + offset2] =
                    accumulator_ptr[step + 16 + 1];
                output_smem[idx2 + (i + 1) * offset4 + offset1] =
                    accumulator_ptr[step + 16 + 2];
                output_smem[idx2 + (i + 1) * offset4 + offset2] =
                    accumulator_ptr[step + 16 + 3];
                step += 4;
            }
        }
        __syncthreads();
        for (int i = 0; i < 16; i++) {
            output_tile[i] = output[smem_idx + i * offset];
            output_tile[i + 16] =
                output[smem_idx + 2 * BN_P * 16 * 4 + i * offset];
        }
        transform_output_tile(result, output_tile, transform_tile, out_size,
                              tile_dim, round, k, out_thread, mask, tile_idx,
                              output_global_offset);
    }
}
