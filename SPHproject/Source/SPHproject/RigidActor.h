// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RigidActor.generated.h"

UCLASS()
class SPHPROJECT_API ARigidActor : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ARigidActor();

	UPROPERTY(BlueprintReadWrite)
		float h;

	UPROPERTY(BlueprintReadWrite)
		float Mass = 5.f;


	UPROPERTY(BlueprintReadWrite)
		TArray<FVector> offsetCM;

	UPROPERTY(BlueprintReadWrite)
		float Momentom = 5000;

	UPROPERTY(BlueprintReadWrite)
		FVector Velocity = FVector(0,0,0);

	UPROPERTY(BlueprintReadWrite)
		FVector Axis = FVector(0, 0, 1);

	UPROPERTY(BlueprintReadWrite)
		float Angle = 0;

	FVector angularVelocity = FVector(0, 0, 0);

	UPROPERTY(BlueprintReadWrite)
		TArray<FVector> SamplePosWS;
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	FVector StartFloor;
	FVector EndTop;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;
	bool moveRigid(FVector totalForce, FVector totalTorque,float dt) {
		Velocity += (totalForce / Mass + FVector(0.f, 0.f, -8.f)) * dt;
		FVector newLoc = Velocity * 100.f * dt + GetActorLocation();

		
		angularVelocity += totalTorque * dt / Momentom;
		
		if (newLoc.X < StartFloor.X ) {
			newLoc.X = StartFloor.X;
			Velocity.X *= -0.4;
		}
		else if (newLoc.X > EndTop.X) {
			newLoc.X = EndTop.X;
			Velocity.X = -0.4;
		}
		if (newLoc.Y < StartFloor.Y) {
			newLoc.Y = StartFloor.Y;
			Velocity.Y = -0.4;
		}
		else if (newLoc.Y > EndTop.Y) {
			newLoc.Y = EndTop.Y;
			Velocity.Y = -0.4;
		}
		if (newLoc.Z < StartFloor.Z) {
			newLoc.Z = StartFloor.Z;
			Velocity.Z = -0.4;
		}
		else if (newLoc.Z > EndTop.Z) {
			newLoc.Z = EndTop.Z;
			Velocity.Z = -0.4;
		}
		SetActorLocation(newLoc);
		float w = angularVelocity.Size();
		if (w > 0.0001) {
			Axis = angularVelocity.GetSafeNormal();
			Angle = w * dt;
			return true;
		}
		return false;
	}


	UFUNCTION(BlueprintCallable, Category = "RigidSPH")
		void SetBoundary(FVector StartFloorWS, FVector EndTopWS, float MaxRadius) {
		StartFloor = StartFloorWS + MaxRadius;
		EndTop = EndTopWS - MaxRadius;
	}


	UFUNCTION(BlueprintCallable, Category = "RigidSPH")
		TArray<FTransform> InitBoundaryParticles2(const FVector vertexPos, const FVector Center, const float X, const float Y, const float Z) {
		TArray <FTransform> Rigidtrf;
		float Hscaled = h * 100;
		float Hscaled2 = h * 0.86 * 100;
		float scale = 0.11f;
		for (int x = 0; x < round(X/ Hscaled2) + 1; x++) {
			for (int y = x % 2; y < round(Y/ Hscaled) + 1; y++) {
				if (x % 2) {
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(x * Hscaled2, y * Hscaled - Hscaled2/2, 0.f),
						FVector(scale, scale, scale)
					));
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(x * Hscaled2, y * Hscaled - Hscaled2/2, Z - 1),
						FVector(scale, scale, scale)
					));
				}
				else {
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(x * Hscaled2, y * Hscaled, 0.f),
						FVector(scale, scale, scale)
					));
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(x * Hscaled2, y * Hscaled, Z - 1),
						FVector(scale, scale, scale)
					));
				}
			}
		}

		for (int x = 0; x < round(X / Hscaled2) + 1; x++) {
			for (int z = 1 - x % 2; z < round(Z / Hscaled) + 1; z++) {
				if (x % 2) {
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(x * Hscaled2, 0.f, z * Hscaled),
						FVector(scale, scale, scale)
					));
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(x * Hscaled2, Y - 1, z * Hscaled),
						FVector(scale, scale, scale)
					));
				}
				else {
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(x * Hscaled2, 0.f, z * Hscaled - Hscaled2 / 2),
						FVector(scale, scale, scale)
					));
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(x * Hscaled2, Y - 1, z * Hscaled - Hscaled2 / 2),
						FVector(scale, scale, scale)
					));
				}
			}
		}

		for (int y = 1; y < round(Y / Hscaled2)  ; y++) {
			for (int z = 1; z < round(Z / Hscaled) + 1 - y%2; z++) {
				if (y % 2) {
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(0.f, y * Hscaled2, z * Hscaled),
						FVector(scale, scale, scale)
					));
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(X -1, y * Hscaled2, z * Hscaled),
						FVector(scale, scale, scale)
					));
				}
				else {
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(0, y * Hscaled2, z * Hscaled - Hscaled2 / 2),
						FVector(scale, scale, scale)
					));
					Rigidtrf.Add(FTransform(
						FRotator(0, 0, 0),
						vertexPos + FVector(X - 1, y * Hscaled2, z * Hscaled - Hscaled2 / 2),
						FVector(scale, scale, scale)
					));
				}
			}
		}

		for (int i = 0; i < Rigidtrf.Num(); i++) {
			offsetCM.Add((Rigidtrf[i].GetTranslation() - Center) / 100.f);
		}

		return Rigidtrf;
	}
};
