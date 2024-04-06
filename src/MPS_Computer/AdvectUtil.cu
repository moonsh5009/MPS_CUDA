#include "stdafx.h"
#include "AdvectUtil.h"
#include "AdvectUtil.cuh"
#include "../MPS_Object/MPSGBArchiver.h"

namespace
{
	constexpr auto nBlockSize = 1024u;
}

#include "thrust/host_vector.h"
void mps::kernel::InitMass(mps::Object* pObj)
{
	pObj->m_mass = thrust::host_vector<REAL>(pObj->GetSize(), 1.0);
}

void mps::kernel::InitForce(const mps::ObjectParam& objParam)
{
	InitForce_kernel << < mcuda::util::DivUp(objParam.GetSize(), nBlockSize), nBlockSize >> >
		(objParam);
}

void mps::kernel::ApplyGravity(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam)
{
	ApplyGravity_kernel << < mcuda::util::DivUp(objParam.GetSize(), nBlockSize), nBlockSize >> >
		(physParam, objParam);
}

void mps::kernel::UpdateVelocity(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam)
{
	UpdateVelocity_kernel << < mcuda::util::DivUp(objParam.GetSize(), nBlockSize), nBlockSize >> >
		(physParam, objParam);
}

void mps::kernel::UpdatePosition(const mps::PhysicsParam& physParam, const mps::ObjectParam& objParam)
{
	UpdatePosition_kernel << < mcuda::util::DivUp(objParam.GetSize(), nBlockSize), nBlockSize >> >
		(physParam, objParam);
}