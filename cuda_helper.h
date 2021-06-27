#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

inline void _checkCudaReturnValue(cudaError_t result, const char* const func, const char* const file, const int line)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, (int)(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit((int)(result));
    }
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaCall(val) _checkCudaReturnValue((val), #val, __FILE__, __LINE__)

