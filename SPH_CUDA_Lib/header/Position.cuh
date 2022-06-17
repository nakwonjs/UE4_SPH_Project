/*==========================================================
File Name: Position.cuh
각 파티클에 가해진 합력을 기반으로 속도와 위치정보를 업데이트
===========================================================*/
#pragma once
#include "Common.cuh"
#include "SphCuda.h"


__global__
void getPostion(float3* dev_pos, float3* dev_vel, float3* dev_force, float* density,const float dt, float3* debug,const unsigned int size ,const ParamSet Param) {
    int stride = gridDim.x * blockDim.x;
    float3 accel;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        accel = dev_force[idx] / density[idx];
        if (idx > Param.activated) {
            dev_vel[idx] = make_float3(0, 0, 0);
            continue;
        }

        dev_vel[idx] = dev_vel[idx] + dt * accel;
        dev_pos[idx] = dev_pos[idx] + dt * dev_vel[idx];

        if (dev_vel[idx].x > Param.MaxSpeed) dev_vel[idx].x = Param.MaxSpeed;
        else if (dev_vel[idx].x < -Param.MaxSpeed) dev_vel[idx].x = -Param.MaxSpeed;

        if (dev_vel[idx].y > Param.MaxSpeed) dev_vel[idx].y = Param.MaxSpeed;
        else if (dev_vel[idx].y < -Param.MaxSpeed) dev_vel[idx].y = -Param.MaxSpeed;

        if (dev_vel[idx].z > Param.MaxSpeed) dev_vel[idx].z = Param.MaxSpeed;
        else if (dev_vel[idx].z < -Param.MaxSpeed) dev_vel[idx].z = -Param.MaxSpeed;
        // 경계조건
        if (dev_pos[idx].x < Param.startFloor.x + Param.r) {
            dev_pos[idx].x = Param.startFloor.x + Param.r;
            dev_vel[idx].x = dev_vel[idx].x * - 0.5;
        }
        else if (dev_pos[idx].x > Param.endTop.x - Param.r) {
            dev_pos[idx].x = Param.endTop.x - Param.r;
            dev_vel[idx].x = dev_vel[idx].x * -0.5;
        }
        if (dev_pos[idx].y < Param.startFloor.y + Param.r) {
            dev_pos[idx].y = Param.startFloor.y + Param.r;
            dev_vel[idx].y = dev_vel[idx].y * -0.5;
        }
        else if (dev_pos[idx].y > Param.endTop.y - Param.r) {
            dev_pos[idx].y = Param.endTop.y - Param.r;
            dev_vel[idx].y = dev_vel[idx].y * -0.5;
        }
        if (dev_pos[idx].z < Param.startFloor.z + Param.r) {
            dev_pos[idx].z = Param.startFloor.z + Param.r;
            dev_vel[idx].z = dev_vel[idx].z * -0.5;
        }
        else if (dev_pos[idx].z > Param.endTop.z - Param.r) {
            dev_pos[idx].z = Param.endTop.z - Param.r;
            dev_vel[idx].z = dev_vel[idx].z * -0.5;
        }
    }
}