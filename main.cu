
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <utility>

#include "config.h"
#include "convolutionForward.cu"
#include "cudnn_runner.cuh"
#include "utils.cuh"

int main(int argc, char *argv[]) {

    const int batch = (argc > 1) ? atoi(argv[1]) : BATCH;
    const int channel = (argc > 2) ? atoi(argv[2]) : CHANNEL;
    const int size = (argc > 3) ? atoi(argv[3]) : SIZE;
    const int Kfilter = (argc > 4) ? atoi(argv[4]) : KFILTER;
    // const int Ksize = (argc > 5) ? atoi(argv[5]) : KSIZE;

    const float TFLOPs = 2.0f * batch * channel * (size + 2 * PAD_H) *
                         (size + 2 * PAD_W) * Kfilter * KSIZE * KSIZE * 1e-12;

    int tiles_dim = ceil(ceil((double)(SIZE + 2) / 2) - 1);
    int elems_dim = tiles_dim * 4;
    // int out_batch = batch, out_channel = Kfilter, out_size = size;

    float *input, *input_cudnn, *filter, *filter_cudnn, *workspace, *output,
        *output_cudnn;
    std::tie(input, input_cudnn, filter, filter_cudnn, workspace, output,
             output_cudnn) = init(batch, channel, size, Kfilter, KSIZE);

    Cudnn_runner runner(batch, channel, size, Kfilter, KSIZE);

    timer("cudnn", TFLOPs, [&runner, input_cudnn, filter_cudnn, output_cudnn] {
        runner.run(input_cudnn, filter_cudnn, output_cudnn);
    });

    timer("my", TFLOPs, [&] {
        convolutionFwd(input, batch, channel, size, filter, Kfilter, KSIZE,
                       output, workspace, ALPHA, tiles_dim);
    });

    return 0;
}