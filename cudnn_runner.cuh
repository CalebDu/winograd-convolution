#pragma once
#include <cudnn.h>

#include "config.h"
#include "utils.cuh"

class Cudnn_runner {
  private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t ipt_desc, out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    size_t workspace_size;
    float *workspace;
    float alpha, beta;
    cudnnConvolutionFwdAlgo_t algo;

  public:
    Cudnn_runner(int batch, int channel, int size, int k, int ksize)
        : alpha(1.0f), beta(0.0f), algo(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD) {

        int out_batch, out_channel, out_size;
        CUDNN_CHECK(cudnnCreate(&cudnn));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&ipt_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(ipt_desc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT, batch, channel,
                                               size, size));

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filt_desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT,
                                               CUDNN_TENSOR_NCHW, k, channel,
                                               ksize, ksize));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
            conv_desc, PAD_H, PAD_W, STR_H, STR_W, DIL_H, DIL_W,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
            conv_desc, ipt_desc, filt_desc, &out_batch, &out_channel, &out_size,
            &out_size));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_batch,
            out_channel, out_size, out_size));
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, ipt_desc, filt_desc, conv_desc, out_desc, algo,
            &workspace_size));

        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    void run(float *input, float *filter, float *output) {
        CUDNN_CHECK(cudnnConvolutionForward(
            cudnn, &alpha, ipt_desc, input, filt_desc, filter, conv_desc, algo,
            workspace, workspace_size, &beta, out_desc, output));
    }
    ~Cudnn_runner() {
        CUDA_CHECK(cudaFree(workspace));
        CUDNN_CHECK(cudnnDestroy(cudnn));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filt_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(ipt_desc));
    }
};