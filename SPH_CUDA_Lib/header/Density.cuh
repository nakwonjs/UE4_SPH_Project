/*==========================================================
File Name: Force.cuh

각 입자의 이웃 입자의 Distance를 기반으로 pressure와 density을 계산


실행 결과: 각 파티클의 pressure, density

===========================================================*/
#pragma once
#include "Common.cuh"
#include "SphCuda.h"


__global__
void getDensity(float3* dev_pos, float* dev_density,float* dev_p,  const int* dev_PID, const int* dev_BinID, const int* dev_PBM, float3* debug, const unsigned int size, const ParamSet Param) {

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

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        PID = dev_PID[idx];
        BinID = dev_BinID[idx];


        pBinCoord.x = BinID % Param.binSize3.x;
        pBinCoord.y = (BinID % (Param.binSize3.x * Param.binSize3.y)) / Param.binSize3.x;
        pBinCoord.z = BinID / (Param.binSize3.x * Param.binSize3.y);

        dev_density[PID] = 0;

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
//==========================================여기서부터 계산관련 코드 작성================================================================================
                    
                    dist = getDistance(dev_pos[PID], dev_pos[j]);
                    
                    if (dist < Param.h) {
                        dev_density[PID] += powf((Param.h * Param.h - dist * dist), 3.f);
                    }
                    
                    //==================================================================================================
                }
            }
        }
        dev_density[PID] *= (Param.Poly6 * Param.mass);
        dev_density[PID] = fmaxf(dev_density[PID], Param.rest_dens);
        dev_p[PID] = Param.gas_const * (dev_density[PID] - Param.rest_dens);

        debug[idx].x = dev_density[PID];


    }
}