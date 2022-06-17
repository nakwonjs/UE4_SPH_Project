// Copyright SCIEMENT, Inc.
// by Hirofumi Seo, M.D., CEO & President

#include "header/SphCuda.h"
#include "header/Common.cuh"
#include "header/Density.cuh"
#include "header/InitParticle.cuh"
#include "header/Position.cuh"
#include "header/FRNN.cuh"
#include "header/Force.cuh"


#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>




cudaError_t SPH_CUDA::initSnowCUDA(unsigned int _size, unsigned int _Max,float3 startFloor , float3 endTop ,std::string* error_message) {
    Max = _Max;
    size = _size;
    Param.RigIdx = size;
    Param.Max = Max;
    Param.size = size;
    Rigidsize = size;
    calcBin(startFloor, endTop);

    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_p, Max * sizeof(float));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc pos failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_density, Max * sizeof(float));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc pos failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_pos, Max * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc pos failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_vel, Max * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc vel failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_force, Max * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc cohForce failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_debug, Max * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc debug failed!";
        postErrorTask();
        return cuda_status;
    }


    cuda_status = cudaMalloc((void**)&dev_PID, Max * sizeof(int));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc PID failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_BinID, Max * sizeof(int));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc BinID failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_Count, (Param.binLen + 1) * sizeof(int));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc BinID failed!";
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&dev_PBM, (Param.binLen + 1) * sizeof(int));
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMalloc BinID failed!";
        postErrorTask();
        return cuda_status;
    }



    initParticles << <GRID_DIM, BLOCK_DIM >> > (dev_vel, Max, Param);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        *error_message = "Kernel initParticles launch failed: " + std::string(cudaGetErrorString(cuda_status));
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "initParticles cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }


}

cudaError_t SPH_CUDA::UpdateRigidSample(float3* SampledPos, float3* SamlpedVel, std::string* error_message) {
    cuda_status = cudaMemcpy(dev_pos + size, SampledPos, Param.RigidSampleSize * sizeof(float3), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy SampledPos H2D failed!";
        postErrorTask();
        return cuda_status;
    }
    cuda_status = cudaMemcpy(dev_vel + size, SamlpedVel, Param.RigidSampleSize * sizeof(float3), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy SampledPos H2D failed!";
        postErrorTask();
        return cuda_status;
    }
    Rigidsize = size + Param.RigidSampleSize;
}

cudaError_t SPH_CUDA::UpdateRigidForce(float3* rigforce, std::string* error_message) {
    cuda_status = cudaMemcpy(rigforce, dev_force + size, Param.RigidSampleSize * sizeof(float3), cudaMemcpyDeviceToHost);//실행결과를 host로 출력
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }
}

cudaError_t SPH_CUDA::StartSimulationCUDA(float3* pos, std::string* error_message) {
    cuda_status = cudaMemcpy(dev_pos, pos, size * sizeof(float3), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy pos H2D failed!";
        postErrorTask();
        return cuda_status;
    }
    return cuda_status;
}


cudaError_t SPH_CUDA::UpdateDensity(float3* debug, std::string* error_message) {

    if (isCrashed) return cuda_status;


    getDensity << <GRID_DIM, BLOCK_DIM >> > (dev_pos,dev_density,dev_p, dev_PID,dev_BinID, dev_PBM ,dev_debug, Rigidsize, Param);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        *error_message = "Kernel launch failed: " + std::string(cudaGetErrorString(cuda_status));
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }


    cuda_status = cudaMemcpy(debug, dev_debug, size * sizeof(float3), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy debug D2H failed!";
        postErrorTask();
        return cuda_status;
    }
}


cudaError_t SPH_CUDA::UpdateForce(float3* debug, std::string* error_message) {

    if (isCrashed) return cuda_status;


    getForce << <GRID_DIM, BLOCK_DIM >> > (dev_pos, dev_vel,dev_density,dev_p,dev_force, dev_PID, dev_BinID, dev_PBM, dev_debug, Rigidsize, Param);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        *error_message = "Kernel launch failed: " + std::string(cudaGetErrorString(cuda_status));
        postErrorTask();
        return cuda_status;
    }

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }


    cuda_status = cudaMemcpy(debug, dev_debug, size * sizeof(float3), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy debug D2H failed!";
        postErrorTask();
        return cuda_status;
    }
}

cudaError_t SPH_CUDA::UpdatePosition(float3* pos,float deltaTime,  std::string* error_message)
{
    if (isCrashed) return cuda_status;
    getPostion << <GRID_DIM, BLOCK_DIM >> > (dev_pos,dev_vel,dev_force,dev_density,deltaTime,dev_debug,size,Param);//커널 실행
    cuda_status = cudaDeviceSynchronize();//디바이스 실행완료 기다리기
    cuda_status = cudaMemcpy(pos, dev_pos, size * sizeof(float3), cudaMemcpyDeviceToHost);//실행결과를 host로 출력
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaMemcpy density D2H failed!";
        postErrorTask();
        return cuda_status;
    }

}






void SPH_CUDA::postErrorTask() {
    cudaFree(dev_p);
    cudaFree(dev_density);
    cudaFree(dev_pos);
    cudaFree(dev_vel);
    cudaFree(dev_force);
    cudaFree(dev_debug);
    isCrashed = true;
}


void SPH_CUDA::calcBin(float3 _startFloor, float3 _endTop) {

    Param.startFloor = _startFloor;
    Param.endTop = _endTop;
    Param.binSize3.x = ceil((_endTop.x - _startFloor.x) / Param.h);
    Param.binSize3.y = ceil((_endTop.y - _startFloor.y) / Param.h);
    Param.binSize3.z = ceil((_endTop.z - _startFloor.z) / Param.h);


    Param.binLen = Param.binSize3.x * Param.binSize3.y * Param.binSize3.z;

}

cudaError_t SPH_CUDA::FRNN(std::string* error_message) {

    if (isCrashed) return cuda_status;



    resetPBM << <GRID_DIM, BLOCK_DIM >> > (dev_PBM, dev_Count, Param.binLen);

    getBinLoc << <GRID_DIM, BLOCK_DIM >> > (dev_pos, dev_BinID, dev_PID, Rigidsize, Param);

    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }

    thrust::device_ptr<int> dev_BinID_ptr(dev_BinID);
    thrust::device_ptr<int> dev_PID_ptr(dev_PID);

    thrust::sort_by_key(dev_BinID_ptr, dev_BinID_ptr + Rigidsize, dev_PID_ptr);


    countElem << <GRID_DIM, BLOCK_DIM >> > (dev_BinID, dev_Count, Rigidsize);


    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        *error_message = "cudaDeviceSynchronize returned error code " + std::to_string(cuda_status) + " after launching addKernel!";
        postErrorTask();
        return cuda_status;
    }


    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, dev_Count, dev_PBM, Param.binLen + 1);
    // Scan 계산을 위한 메모리를 확인 

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // 메모리 할당

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, dev_Count, dev_PBM, Param.binLen + 1);
    // Scan 계산

    cudaFree(d_temp_storage);
    // Scan 계산용 임시 메모리 해제



    return cuda_status;

}