#pragma once
// #ifndef UTILS_HEADER
// #define UTILS_HEADER

#include <curand.h>
#include <curand_kernel.h>
#include <functional>
#include <iostream>
#include <tuple>
#include <utility>

#define CUDA_CHECK(f)                                                          \
    {                                                                          \
        cudaError_t err = (f);                                                 \
        if (err != cudaSuccess) {                                              \
            std::cout << "    Error occurred: " << err << std::endl;           \
            std::exit(1);                                                      \
        }                                                                      \
    }

#define CUDNN_CHECK(f)                                                         \
    {                                                                          \
        cudnnStatus_t err = (f);                                               \
        if (err != CUDNN_STATUS_SUCCESS) {                                     \
            printf("    Error occurred: \n");                                  \
            std::exit(1);                                                      \
        }                                                                      \
    }

// function copy access matrix to gpu mem
void access_mat_cpy() {
    CUDA_CHECK(cudaMemcpyToSymbol(access_filter, aux, sizeof(aux)));
    CUDA_CHECK(cudaMemcpyToSymbol(access_input, aux2, sizeof(aux2)));
    CUDA_CHECK(cudaMemcpyToSymbol(access_out, aux3, sizeof(aux3)));
    CUDA_CHECK(cudaMemcpyToSymbol(out_thread, aux4, sizeof(aux4)));
}

__global__ void rand_init_data(float *arr, int len, long long seed = 1) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= len) {
        return;
    }
    curandState_t state;
    curand_init(seed, tid, 0LL, &state);
    arr[tid] = curand_uniform(&state);
}
// nchw -> chwn
__global__ void dataCpy(float *dst, float *src, int batch, int channel,
                        int size) {
    // auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    // if (tid >= len) {
    //     return;
    // }
    // dst[tid] = src[tid];
    int src_idx = blockIdx.y + blockIdx.z * size + threadIdx.x * size * size +
                  blockIdx.x * size * size * channel;
    int dst_idx = blockIdx.x + blockIdx.y + batch + blockIdx.z * batch * size +
                  threadIdx.x * batch * size * size;
    dst[dst_idx] = src[src_idx];
}

void result_checker(float *ours, float *cudnn, int batch, int size, int channel,
                    int shift) {
    int error_cnt = 0;
    float max_error = 0.0f;
    for (int c = 0; c < channel; c++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int n = 0; n < batch; n++) {
                    auto diff = fabs(ours[n + j * batch + i * batch * size +
                                          c * batch * size * size] -
                                     cudnn[j + i * size + c * size * size +
                                           n * size * size * channel]);
                    if (diff > 1) {
                        error_cnt++;
                        printf("error in h: %d, w: %d, c: %d, n: %d, ours: %f "
                               "vs cudnn: %f",
                               i, j, c, n,
                               ours[n + j * batch + i * batch * size +
                                    c * batch * size * size],
                               diff);
                    }
                    max_error = fmax(diff, max_error);
                }
            }
        }
    }
    printf("max error: %f, error_cnt: %d", max_error, error_cnt);
}

cudaError_t init_all_data(float *input, float *input_cudnn, float *filter,
                          float *filter_cudnn, int batch, int channel, int size,
                          int k, int ksize) {
    auto n = batch * channel * size * size;
    auto thread = 256;
    dim3 grid((n - 1) / thread + 1);
    rand_init_data<<<grid, thread>>>(input_cudnn, n);
    dataCpy<<<dim3(batch, size, size), channel>>>(input, input_cudnn, batch,
                                                  channel, size);

    n = k * channel * ksize * ksize;
    grid = dim3((n - 1) / thread + 1);
    rand_init_data<<<grid, thread>>>(filter_cudnn, n);
    dataCpy<<<dim3(k, ksize, ksize), channel>>>(filter, filter_cudnn, k,
                                                channel, ksize);
    return cudaGetLastError();
}

void Tlops(int batch, int channel, int size, int k, int ksize, int pad,
           float timer) {
    double tflops = static_cast<double>(1) * 2.0f * batch * channel *
                    (size + 2 * pad) * (size + 2 * pad) * k * ksize * ksize /
                    (2.25 * timer * 1e9);
    printf("%.3f ms, %.2f tflops", timer, tflops);
}

decltype(auto) init(const int batch, const int channel, const int size,
                    const int k, const int ksize) {
    constexpr size_t Width = sizeof(float);

    // initialize device space for each component
    size_t input_Nbyte = batch * channel * size * size * Width;
    size_t filter_Nbyte = k * channel * ksize * ksize * Width;
    size_t workspace_Nbyte = k * channel * 4 * 4 * Width;
    size_t output_Nbyte = k * batch * size * size * Width;
    float *input, *input_cudnn, *filter, *filter_cudnn, *workspace, *output,
        *output_cudnn;

    // malloc device memory
    CUDA_CHECK(cudaMalloc(&input, input_Nbyte));
    CUDA_CHECK(cudaMalloc(&input_cudnn, input_Nbyte));
    CUDA_CHECK(cudaMalloc(&filter, filter_Nbyte));
    CUDA_CHECK(cudaMalloc(&filter_cudnn, filter_Nbyte));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_Nbyte));
    CUDA_CHECK(cudaMalloc(&output, output_Nbyte));
    CUDA_CHECK(cudaMalloc(&output_cudnn, output_Nbyte));
    init_all_data(input, input_cudnn, filter, filter_cudnn, batch, channel,
                  size, k, ksize);
    // copy access matrix to device 
    access_mat_cpy();
    
    return std::make_tuple(input, input_cudnn, filter, filter_cudnn, workspace,
                           output, output_cudnn);
}
void timer(const std::string &tag, const float tflop,
           const std::function<void()> &f, int rounds = 10) {
    // warmup
    f();
    // warmup finished
    float round;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    (cudaEventRecord(start));
    for (int i = 0; i < rounds; i++) {
        f();
    }
    (cudaEventRecord(end));
    (cudaEventSynchronize(end));
    (cudaEventElapsedTime(&round, start, end));
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    round /= rounds;
    auto tflops = tflop / (2.25 * round * 1e-3);
    printf("%s version: %.2f TFLOPS in %d round\n", tag.c_str(), tflops,
           rounds);
}
// #endif