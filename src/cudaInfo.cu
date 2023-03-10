// Logger Lib
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/spdlog.h>

#include "cudaInfo.cuh"

bool checkCuda() {
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0) {
        SPDLOG_ERROR("Error: checkCuda: No CUDA devices found");
        return false;
    }
    for (auto i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        SPDLOG_INFO(spdlog::fmt_lib::format("Cuda Device {}", prop.name));
        SPDLOG_INFO(spdlog::fmt_lib::format("\t├ Compute Units {}", prop.multiProcessorCount));
        SPDLOG_INFO(spdlog::fmt_lib::format("\t├ Max Work Group Size {}", prop.warpSize));
        SPDLOG_INFO(spdlog::fmt_lib::format("\t├ Local Mem Size {}", prop.sharedMemPerBlock));
        SPDLOG_INFO(spdlog::fmt_lib::format("\t└ Global Mem Size {}", prop.totalGlobalMem));
    }
    return true;
}