#pragma once

#include <string>
#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.h"
#include "device_launch_parameters.h"

#define GRID_DIM 32
#define BLOCK_DIM 128
#define MAX_OF_COLLIDER 10




struct ParamSet {
    
    float h;
    float r;
    float rest_dens;
    float mass;
    float visc;
    float gas_const;

    float Poly6;
    float Spick_GradD;
    float Visc_Lap;

    float gravity;
    float Psi;

    float3 startFloor;
    float3 endTop;
	float3 rootPos;

    int3 binSize3;

    float MaxSpeed = 2.f;
    unsigned int binLen;
    unsigned int Max;
    unsigned int size;
    unsigned int RigIdx;
    int activated = 0;
	int RigidSampleSize = 0;
};


class SPH_CUDA {
private:
    void calcBin(float3 startFloor, float3 endTop);

public:
    float* dev_p = 0;
    float* dev_density = 0;
    float3* dev_pos = 0;
    float3* dev_vel = 0;
    float3* dev_force = 0;
    float3* dev_debug = 0;
    ParamSet Param;


    // FRNN 관련 변수
    int* dev_PID;
    int* dev_BinID;
    int* dev_Count; // BinLen + 1 
    int* dev_PBM; // BinLen + 1 


    // 강체관련 변수

    

    unsigned int Max;
    unsigned int size;
    unsigned int Rigidsize;

    cudaError_t cuda_status;
    bool isCrashed = false;

    SPH_CUDA() {
	}
    ~SPH_CUDA() {}
    cudaError_t initSnowCUDA(unsigned int _size, unsigned int _MAX,  float3 startFloor, float3 endTop, std::string* error_message);
    cudaError_t UpdateDensity(float3* debug, std::string* error_message);

    cudaError_t UpdatePosition(float3* acc,float deltaTime,  std::string* error_message);
    cudaError_t UpdateForce(float3* debug, std::string* error_message);
    cudaError_t FRNN(std::string* error_message);
    void postErrorTask();
    cudaError_t UpdateRigidForce(float3* rigforce, std::string* error_message);
	
	cudaError_t StartSimulationCUDA(float3* Pos, std::string* error_message);
	cudaError_t UpdateRigidSample(float3* SampledPos, float3* SamlpedVel, std::string* error_message);
    
	
};
