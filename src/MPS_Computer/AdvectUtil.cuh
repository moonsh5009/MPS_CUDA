#include "AdvectUtil.h"
#include "../MCUDA_Lib/MCUDAHelper.cuh"
#include "../MPS_Object/MPSUniformDef.h"

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