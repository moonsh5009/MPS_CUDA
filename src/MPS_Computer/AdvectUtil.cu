#include "stdafx.h"
#include "AdvectUtil.cuh"

#include "../MPS_Object/MPSGBArchiver.h"

namespace
{
	constexpr auto nBlockSize = 1024u;
}

#include "thrust/host_vector.h"
#include "AdvectUtil.h"
void mps::kernel::InitMass(mps::Object* pObj)
{
	pObj->m_mass = thrust::host_vector<REAL>(pObj->GetSize(), 1.0);
}

void mps::kernel::ResetForce(const mps::ObjectParam& objParam)
{
	const auto nSize = objParam.GetSize();
	if (nSize == 0) return;

	thrust::fill(thrust::device_ptr<REAL3>(objParam.GetForceArray()), thrust::device_ptr<REAL3>(objParam.GetForceArray() + objParam.GetSize()),
		REAL3{ 0.0 });
}

void mps::kernel::ApplyGravity(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam)
{
	const auto nSize = objParam.GetSize();
	if (nSize == 0) return;

	ApplyGravity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(physParam, objParam);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::UpdateVelocity(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam)
{
	const auto nSize = objParam.GetSize();
	if (nSize == 0) return;

	UpdateVelocity_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(physParam, objParam);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::UpdatePosition(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam)
{
	const auto nSize = objParam.GetSize();
	if (nSize == 0) return;

	UpdatePosition_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(physParam, objParam);
	CUDA_CHECK(cudaPeekAtLastError());
}

void mps::kernel::BoundaryCollision(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam)
{
	const auto nSize = objParam.GetSize();
	if (nSize == 0) return;

	BoundaryCollision_kernel << < mcuda::util::DivUp(nSize, nBlockSize), nBlockSize >> >
		(physParam, objParam);
	CUDA_CHECK(cudaPeekAtLastError());
}
