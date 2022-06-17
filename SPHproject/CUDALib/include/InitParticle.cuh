#pragma once
#include "Common.cuh"
#include "SphCuda.h"


__global__
void initParticles(float3* dev_velocity, const unsigned int size, ParamSet Param) {
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
            dev_velocity[idx] = make_float3(0.f, 0.f, 0.f);
    }
}
