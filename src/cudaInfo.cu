//
// Created by tztz8 on 3/10/23.
//
#include <stdio.h>
#include <stdlib.h>

#include "cudaInfo.cuh"

bool checkCuda() {
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0) {
        fprintf(stderr, "Error: checkCuda: No CUDA devices found\n");
        return false;
    }
    for (auto i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Debug: Cuda Device %s\n", prop.name);
        printf("Debug: \t├ Compute Units %d\n", prop.multiProcessorCount);
        printf("Debug: \t├ Max Work Group Size %d\n", prop.warpSize);
        printf("Debug: \t├ Local Mem Size %zu\n", prop.sharedMemPerBlock);
        printf("Debug: \t└ Global Mem Size %zu\n", prop.totalGlobalMem);
    }
    return true;
}