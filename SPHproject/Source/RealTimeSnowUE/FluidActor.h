// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "SphCuda.h"
#include "GameFramework/Actor.h"
#include "cuda_runtime.h"
#include "RigidActor.h"
#include "FluidActor.generated.h"

#define AMOUNT 20000
#define MAX_RIG 1024

UCLASS()
class SPHPROJECT_API AFluidActor : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AFluidActor();

	SPH_CUDA SPH = SPH_CUDA();

	UPROPERTY(BlueprintReadWrite)
		TArray<ARigidActor*> rigidArr;

	float3 Pos[AMOUNT];

	float3 RigidSamplePos[MAX_RIG];
	float3 RigidSampleVel[MAX_RIG];
	float3 RigidSampleForce[MAX_RIG];

	float3 debug[AMOUNT];

	UPROPERTY(BlueprintReadWrite)
		int Amount = AMOUNT;

	double st;
	double ed;
	UPROPERTY(BlueprintReadWrite)
		TArray<FTransform> ParticleTrfs;

	UPROPERTY(BlueprintReadWrite)
		TArray<FVector> SnowArrM;

	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool InitSPH(FVector StartFloor, FVector EndTop, FVector rootPos, int InitAmount);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdateDensity();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdateForce();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdatePosition(float DeltaTime);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		void FRNN();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		void StartSimulation();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		bool UpdateRigForce(float dt);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		void UpdateCollider();
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		void AddCollider(ARigidActor* rigidActor);
	UFUNCTION(BlueprintCallable, Category = "SnowCUDA")
		void AddActivated(int n);
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
