#pragma once
#include "AdvectUtil.h"

#include "../MCUDA_Lib/MCUDAHelper.cuh"
#include "../MPS_Object/MPSUniformDef.h"
#include "../MPS_Object/MPSObject.h"

__global__ void InitForce_kernel(const mps::ObjectParam objParam)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objParam.GetSize()) return;

	objParam.Force(id) = REAL3{ 0.0 };
}
__global__ void ApplyGravity_kernel(const mps::PhysicsParam physicsParam, const mps::ObjectParam objParam)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objParam.GetSize()) return;

	objParam.Force(id) += physicsParam.gravity * objParam.Mass(id);
}
__global__ void UpdateVelocity_kernel(const mps::PhysicsParam physicsParam, const mps::ObjectParam objParam)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objParam.GetSize()) return;

	const auto mass = objParam.Mass(id);
	if (mass > 1.0e-10)
	{
		objParam.Velocity(id) += physicsParam.dt / mass * objParam.Force(id);
	}
}

__global__ void UpdatePosition_kernel(const mps::PhysicsParam physicsParam, const mps::ObjectParam objParam)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objParam.GetSize()) return;

	objParam.Position(id) += physicsParam.dt * objParam.Velocity(id);
}

__global__ void BoundaryCollision_kernel(const mps::PhysicsParam physicsParam, const mps::ObjectParam objParam)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= objParam.GetSize()) return;

	auto vel = objParam.Velocity(id);
	const auto pos = objParam.Position(id) + vel * physicsParam.dt;

	if (pos.x < physicsParam.min.x && vel.x < 0.) vel.x = (abs(vel.x) < 1.0e-4) ? 0.0 : -vel.x * 0.4;
	else if (pos.x > physicsParam.max.x && vel.x > 0.) vel.x = (abs(vel.x) < 1.0e-4) ? 0.0 : -vel.x * 0.4;
	if (pos.y < physicsParam.min.y && vel.y < 0.) vel.y = (abs(vel.y) < 1.0e-4) ? 0.0 : -vel.y * 0.4;
	else if (pos.y > physicsParam.max.y && vel.y > 0.) vel.y = (abs(vel.y) < 1.0e-4) ? 0.0 : -vel.y * 0.4;
	if (pos.z < physicsParam.min.z && vel.z < 0.) vel.z = (abs(vel.z) < 1.0e-4) ? 0.0 : -vel.z * 0.4;
	else if (pos.z > physicsParam.max.z && vel.z > 0.) vel.z = (abs(vel.z) < 1.0e-4) ? 0.0 : -vel.z * 0.4;

	/*const auto pos = objParam.Position(id);
	auto vel = objParam.Velocity(id);
	for (int i = 0; i < 3; i++)
	{
		if (pos[i] + vel[i] * physicsParam.dt < physicsParam.min[i])
			vel[i] = (physicsParam.min[i] - pos[i]) / physicsParam.dt;
		else if (pos[i] + vel[i] * physicsParam.dt > physicsParam.max[i])
			vel[i] = (physicsParam.max[i] - pos[i]) / physicsParam.dt;
	}*/
	objParam.Velocity(id) = vel;
}