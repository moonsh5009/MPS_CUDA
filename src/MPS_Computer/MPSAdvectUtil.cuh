#pragma once

#include "MPSAdvectUtil.h"

#include "../MCUDA_Lib/MCUDAHelper.cuh"
#include "../MPS_Object/MPSUniformDef.h"

__global__ void ApplyGravity_kernel(
	mps::PhysicsParam physicsParam,
	REAL3* MCUDA_RESTRICT pObjForce,
	const REAL* MCUDA_RESTRICT pObjMass,
	size_t objSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	pObjForce[id] += physicsParam.gravity * pObjMass[id];
}

__global__ void UpdateVelocity_kernel(
	mps::PhysicsParam physicsParam,
	REAL3* MCUDA_RESTRICT pObjVelocity,
	const REAL3* MCUDA_RESTRICT pObjForce,
	const REAL* MCUDA_RESTRICT pObjMass,
	size_t objSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	if (pObjMass[id] > 1.0e-10)
	{
		pObjVelocity[id] += physicsParam.dt / pObjMass[id] * pObjForce[id];
	}
}

__global__ void UpdatePosition_kernel(
	mps::PhysicsParam physicsParam,
	REAL3* MCUDA_RESTRICT pObjPosition,
	const REAL3* MCUDA_RESTRICT pObjVelocity,
	size_t objSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	pObjPosition[id] += physicsParam.dt * pObjVelocity[id];
}

__global__ void BoundaryCollision_kernel(
	mps::PhysicsParam physicsParam,
	const REAL3* MCUDA_RESTRICT pObjPosition,
	REAL3* MCUDA_RESTRICT pObjVelocity,
	size_t objSize)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objSize) return;

	auto vel = pObjVelocity[id];
	const auto pos = pObjPosition[id] + vel * physicsParam.dt;

	if (pos.x < physicsParam.min.x && vel.x < 0.) vel.x = (abs(vel.x) < 1.0e-4) ? 0.0 : -vel.x * 0.4;
	else if (pos.x > physicsParam.max.x && vel.x > 0.) vel.x = (abs(vel.x) < 1.0e-4) ? 0.0 : -vel.x * 0.4;
	if (pos.y < physicsParam.min.y && vel.y < 0.) vel.y = (abs(vel.y) < 1.0e-4) ? 0.0 : -vel.y * 0.4;
	else if (pos.y > physicsParam.max.y && vel.y > 0.) vel.y = (abs(vel.y) < 1.0e-4) ? 0.0 : -vel.y * 0.4;
	if (pos.z < physicsParam.min.z && vel.z < 0.) vel.z = (abs(vel.z) < 1.0e-4) ? 0.0 : -vel.z * 0.4;
	else if (pos.z > physicsParam.max.z && vel.z > 0.) vel.z = (abs(vel.z) < 1.0e-4) ? 0.0 : -vel.z * 0.4;

	/*const auto pos = pObjPosition[id] ;
	auto vel = pObjVelocity[id];
	for (int i = 0; i < 3; i++)
	{
		if (pos[i] + vel[i] * physicsParam.dt < physicsParam.min[i])
			vel[i] = (physicsParam.min[i] - pos[i]) / physicsParam.dt;
		else if (pos[i] + vel[i] * physicsParam.dt > physicsParam.max[i])
			vel[i] = (physicsParam.max[i] - pos[i]) / physicsParam.dt;
	}*/
	pObjVelocity[id] = vel;
}