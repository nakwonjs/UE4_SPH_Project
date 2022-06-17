/*==========================================================
File Name: CohesionForce.cuh

각 입자에 작용하는 응집력 계산

미리 계산되어야 하는것: 각 파티클의 neighborCounts값
실행 결과: 각 파티클의 coh_Force, positiveForce, negativeForce값 업데이트

===========================================================*/
#pragma once
#include "Common.cuh"
#include "SphCuda.h"


__global__
void getForce(float3* dev_pos,float3* dev_vel, float* dev_density, float* dev_p, float3* dev_force, const int* dev_PID, const int* dev_BinID, const int* dev_PBM, float3* debug, const unsigned int size, const ParamSet Param) {

    int stride = gridDim.x * blockDim.x;
    //FRNN을 위한 변수
    int PID;
    int BinID;
    int start;
    int end;
    int start_x;
    int end_x;
    int new_y;
    int new_z;
    int j;
    int3 pBinCoord;

    float dist;
    float3 pressure;
    float3 viscosity;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        PID = dev_PID[idx];
        BinID = dev_BinID[idx];

        pBinCoord.x = BinID % Param.binSize3.x;
        pBinCoord.y = (BinID % (Param.binSize3.x * Param.binSize3.y)) / Param.binSize3.x;
        pBinCoord.z = BinID / (Param.binSize3.x * Param.binSize3.y);

        pressure = make_float3(0.f,0.f,0.f);
        viscosity = make_float3(0.f, 0.f, 0.f);

        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {

                start_x = max(pBinCoord.x - 1, 0);
                end_x = min(pBinCoord.x + 1, Param.binSize3.x - 1);
                new_y = pBinCoord.y + y;
                new_z = pBinCoord.z + z;
                if (new_y >= Param.binSize3.y || new_y < 0) {
                    continue;
                }
                if (new_z >= Param.binSize3.z || new_z < 0) {
                    continue;
                }
                start = start_x + new_y * Param.binSize3.x + new_z * Param.binSize3.x * Param.binSize3.y;
                end = end_x + new_y * Param.binSize3.x + new_z * Param.binSize3.x * Param.binSize3.y;

                for (int idxP = dev_PBM[start]; idxP < dev_PBM[end + 1]; idxP++) {
                    j = dev_PID[idxP];
                    if (j >= Param.RigIdx && PID >= Param.RigIdx) continue;
                    //==========================================여기서부터 계산관련 코드 작성================================================================================
                    dist = getDistance(dev_pos[PID], dev_pos[j]);

                    if (dist < Param.h) {
                        pressure = pressure - (dev_p[PID] / powf(dev_density[PID], 2.f) + dev_p[j] / powf(dev_density[j], 2.f)) * getNormalVec(dev_pos[PID], dev_pos[j]) * powf(Param.h - dist, 2.f);
                        viscosity = viscosity - (dev_vel[PID] - dev_vel[j]) / dev_density[j] * (Param.h - dist);
                    }

                    //==================================================================================================
                }
            }
        }
        if (PID >= Param.RigIdx) {
            dev_force[PID] = dev_density[PID] * Param.mass * Param.Spick_GradD * pressure + Param.mass * Param.Visc_Lap * Param.visc * viscosity; 
        }
        else{
            dev_force[PID] = dev_density[PID] * Param.mass * Param.Spick_GradD * pressure + Param.mass * Param.Visc_Lap * Param.visc * viscosity + Param.rest_dens * make_float3(0.f, 0.f, -Param.gravity);
        }
        
        debug[idx] = dev_force[PID];


    }
}