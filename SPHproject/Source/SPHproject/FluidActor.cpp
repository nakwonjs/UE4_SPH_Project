// Fill out your copyright notice in the Description page of Project Settings.


#include "FluidActor.h"

// Sets default values
AFluidActor::AFluidActor()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AFluidActor::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AFluidActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

UFUNCTION()
bool AFluidActor::InitSPH(FVector StartFloor, FVector EndTop, FVector rootPos, int InitAmount) {


	SPH.Param.r = 0.05f;
	SPH.Param.h = 2.0f * SPH.Param.r;
	SPH.Param.rest_dens = 1000.0f;
	SPH.Param.mass = powf(SPH.Param.r, 3.0f) * 4.2f * SPH.Param.rest_dens;
	SPH.Param.visc = 3.6f;
	SPH.Param.gas_const = 1.4f;

	SPH.Param.Poly6 = 315.f / (64.f * 3.14f * powf(SPH.Param.h, 9));
	SPH.Param.Spick_GradD = -45.f / (3.14f * powf(SPH.Param.h, 6));
	SPH.Param.Visc_Lap = 45.f / (3.14f * powf(SPH.Param.h, 6));
	SPH.Param.rootPos = make_float3(rootPos.X, rootPos.Y, rootPos.Z);
	SPH.Param.activated = InitAmount;
	SPH.Param.gravity = 5.8f;
	std::string error_message;
	SPH.Param.MaxSpeed = 10.0;
	float3 fStartFloor = make_float3(StartFloor.X / 100.f, StartFloor.Y / 100.f, StartFloor.Z / 100.f);
	float3 fEndTop = make_float3(EndTop.X / 100.f, EndTop.Y / 100.f, EndTop.Z / 100.f);

	cudaError_t cuda_status = SPH.initSnowCUDA(Amount, AMOUNT + MAX_RIG, fStartFloor, fEndTop, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("CUDAInit failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}

	return true;
}


bool AFluidActor::UpdateDensity() {

	std::string error_message;
	cudaError_t cuda_status = SPH.UpdateDensity(debug, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("CUDA Cohesion failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}

	return true;
}

bool AFluidActor::UpdateForce() {
	std::string error_message;
	cudaError_t cuda_status = SPH.UpdateForce(debug, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("CUDA Force failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}

	return true;
}

bool AFluidActor::UpdatePosition(float DeltaTime) {

	std::string error_message;
	cudaError_t cuda_status = SPH.UpdatePosition(Pos, DeltaTime, &error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("Moving Update failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
		return false;
	}

	for (int i = 0; i < ParticleTrfs.Num(); i++)
	{
		ParticleTrfs[i].SetLocation(FVector(
			Pos[i].x * 100.f,
			Pos[i].y * 100.f,
			Pos[i].z * 100.f));
	}

	//ed = FPlatformTime::Seconds();
	//UE_LOG(LogTemp, Warning, TEXT("Pos dt : %f\n\n"), ed - st);
	return true;
}




void AFluidActor::FRNN() {

	std::string error_message;



	cudaError_t cuda_status = SPH.FRNN(&error_message);
	if (cuda_status != cudaSuccess) {
		UE_LOG(LogTemp, Warning, TEXT("FRNN failed!\n"));
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString(error_message.c_str()));
	}


}



void AFluidActor::StartSimulation() {
	std::string error_message;
	for (int i = 0; i < ParticleTrfs.Num(); i++) {
		Pos[i] = make_float3(ParticleTrfs[i].GetLocation().X / 100.f, ParticleTrfs[i].GetLocation().Y / 100.f, ParticleTrfs[i].GetLocation().Z / 100.f);
	}
	cudaError_t cuda_status = SPH.StartSimulationCUDA(Pos, &error_message);
}



void AFluidActor::AddCollider(ARigidActor* rigidActor) {
	std::string error_message;
	rigidArr.Add(rigidActor);
	for (int i = SPH.Param.RigidSampleSize; i < SPH.Param.RigidSampleSize + rigidActor->SamplePosWS.Num(); i++) {
		RigidSamplePos[i] = make_float3(rigidActor->SamplePosWS[i].X / 100.f, rigidActor->SamplePosWS[i].Y / 100.f, rigidActor->SamplePosWS[i].Z / 100.f);
		RigidSampleVel[i] = make_float3(0, 0, 0);
	}
	SPH.Param.RigidSampleSize += rigidActor->SamplePosWS.Num();
	cudaError_t cuda_status = SPH.UpdateRigidSample(RigidSamplePos, RigidSampleVel, &error_message);
}

bool AFluidActor::UpdateRigForce(float dt) {
	std::string error_message;
	cudaError_t cuda_status = SPH.UpdateRigidForce(RigidSampleForce, &error_message);
	for (int i = 0; i < rigidArr.Num(); i++) {
		FVector TotalForce = FVector(0.f, 0.f, 0.f);
		FVector TotalTorque = FVector(0.f, 0.f, 0.f);
		for (int j = 0; j < rigidArr[i]->offsetCM.Num(); j++) {
			FVector Force = FVector(RigidSampleForce[i * j + j].x, RigidSampleForce[i * j + j].y, RigidSampleForce[i * j + j].z);
			TotalForce += Force;
			TotalTorque = FVector::CrossProduct(rigidArr[i]->offsetCM[j], Force);
		}
		return rigidArr[i]->moveRigid(TotalForce, TotalTorque, dt);

	}
	return false;
}

void  AFluidActor::UpdateCollider() {
	std::string error_message;
	for (int i = 0; i < rigidArr.Num(); i++) {
		FVector VelSmaple = FVector(0.f, 0.f, 0.f);

		for (int j = 0; j < rigidArr[i]->offsetCM.Num(); j++) {
			VelSmaple = rigidArr[i]->Velocity + FVector::CrossProduct(rigidArr[i]->offsetCM[j] , rigidArr[i]->angularVelocity);
			RigidSamplePos[i * j + j] = make_float3(rigidArr[i]->SamplePosWS[j].X / 100.f, rigidArr[i]->SamplePosWS[j].Y / 100.f, rigidArr[i]->SamplePosWS[j].Z / 100.f);
			RigidSampleVel[i * j + j] = make_float3(VelSmaple.X / 100.f, VelSmaple.Y / 100.f, VelSmaple.Z / 100.f);
		}
		cudaError_t cuda_status = SPH.UpdateRigidSample(RigidSamplePos, RigidSampleVel, &error_message);

	}
}

void AFluidActor::AddActivated(int n) {
	if (SPH.Param.activated < AMOUNT) {
		SPH.Param.activated += n;
	}
}