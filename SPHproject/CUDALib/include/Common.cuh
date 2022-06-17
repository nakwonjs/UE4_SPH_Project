#pragma once
#include <string>
#include "cuda_runtime.h"

__device__
float getDistance(const float3& pointA, const float3& pointB) {
    return sqrtf(powf((pointA.x - pointB.x), 2.f) + powf((pointA.y - pointB.y), 2.f) + powf((pointA.z - pointB.z), 2.f));
}

__device__
float3 getNormalVec(const float3& ri, const float3& rj) {
    float sq = getDistance(ri, rj);
    if (sq == 0) return make_float3(0.f, 0.f, 0.f); //division 0
    return make_float3((ri.x - rj.x) / sq, (ri.y - rj.y) / sq, (ri.z - rj.z) / sq);

}

__device__
float dot(const float3& f1, const float3& f2) {
    return f1.x * f2.x + f1.y * f2.y + f1.z + f2.z;
}
__device__
float3 operator*(const float& f, const float3& f3) {
    return make_float3(f * f3.x, f * f3.y, f * f3.z);
}

__device__
float3 operator*(const float3& f3, const float& f) {
    return make_float3(f * f3.x, f * f3.y, f * f3.z);
}


__device__
float3 operator/(const float3& f3, const float& f) {
    return make_float3(f3.x / f, f3.y / f, f3.z / f);
}

__device__
float3 operator+(const float3& a, const float3& b) {

    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


__device__
float3 operator-(const float3& a, const float3& b) {

    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
