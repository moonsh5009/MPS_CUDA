#include "stdafx.h"
#include "MPSAdvectUtil.cuh"

#include <thrust/device_vector.h>

#include "../MPS_Object/MPSGBArchiver.h"
#include "../MPS_Object/MPSObjectParam.h"

namespace
{
	constexpr auto nBlockSize = 1024u;
}

void mps::kernel::Advect::ResetREAL(REAL* ptr, size_t size)
{
	if (size == 0) return;

	thrust::fill(thrust::device_pointer_cast(ptr), thrust::device_pointer_cast(ptr + size), 0.0);
}

void mps::kernel::Advect::ResetForce(const mps::ObjectParam& obj)
{
	if (obj.size == 0) return;

	thrust::fill(thrust::device_pointer_cast(obj.pForce), thrust::device_pointer_cast(obj.pForce + obj.size), REAL3{ 0.0 });
}

void mps::kernel::Advect::ApplyGravity(const mps::PhysicsParam& physParam, const mps::ObjectParam& obj)
{
	if (obj.size == 0) return;
	
	ApplyGravity_kernel << < mcuda::util::DivUp(obj.size, nBlockSize), nBlockSize >> > (
		physParam,
		obj.pForce,
		obj.pMass,
		obj.size);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::Advect::UpdateVelocity(const mps::PhysicsParam& physParam, const mps::ObjectParam& obj)
{
	if (obj.size == 0) return;

	UpdateVelocity_kernel << < mcuda::util::DivUp(obj.size, nBlockSize), nBlockSize >> > (
		physParam,
		obj.pVelocity,
		obj.pForce,
		obj.pMass,
		obj.size);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::Advect::UpdatePosition(const mps::PhysicsParam& physParam, const mps::ObjectParam& obj)
{
	if (obj.size == 0) return;

	UpdatePosition_kernel << < mcuda::util::DivUp(obj.size, nBlockSize), nBlockSize >> >(
		physParam,
		obj.pPosition,
		obj.pVelocity,
		obj.size);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::Advect::BoundaryCollision(const mps::PhysicsParam& physParam, const mps::ObjectParam& obj)
{
	if (obj.size == 0) return;

	BoundaryCollision_kernel << < mcuda::util::DivUp(obj.size, nBlockSize), nBlockSize >> >(
		physParam,
		obj.pPosition,
		obj.pVelocity,
		obj.size);
	CUDA_CHECK(cudaPeekAtLastError());
}
