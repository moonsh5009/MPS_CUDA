#include "stdafx.h"
#include "MPSSPHObject.h"

mps::SPHObject::SPHObject() : mps::ParticleObject{}
{
}

void mps::SPHObject::Clear()
{
	mps::ParticleObject::Clear();

	m_density.clear();
	m_pressure.clear();
	m_factorDFSPH.clear();
	m_factorST.clear();
	m_factorSTB.clear();

	m_tempReal.clear();
	m_tempVec3.clear();
	m_tempMat3.clear();
	m_previousVel.clear();
	m_predictVel.clear();
}

void mps::SPHObject::Resize(const size_t size)
{
	mps::ParticleObject::Resize(size);

	m_density.resize(size);
	m_pressure.resize(size);
	m_factorDFSPH.resize(size);
	m_factorST.resize(size);
	m_factorSTB.resize(size);
	
	m_tempReal.resize(size);
	m_tempVec3.resize(size);
	m_tempMat3.resize(size);
	m_previousVel.resize(size);
	m_predictVel.resize(size);
}

std::shared_ptr<mps::ObjectResource> mps::SPHObject::GenerateDeviceResource()
{
	auto pSuperParam = std::static_pointer_cast<mps::ParticleResource>(ParticleObject::GenerateDeviceResource());
	if (!pSuperParam) return {};

	return std::make_shared<SPHResource>(
		pSuperParam,
		thrust::raw_pointer_cast(m_density.data()),
		thrust::raw_pointer_cast(m_pressure.data()),
		thrust::raw_pointer_cast(m_factorDFSPH.data()),
		thrust::raw_pointer_cast(m_factorST.data()),
		thrust::raw_pointer_cast(m_factorSTB.data()),
		thrust::raw_pointer_cast(m_tempReal.data()),
		thrust::raw_pointer_cast(m_tempVec3.data()),
		thrust::raw_pointer_cast(m_tempMat3.data()),
		thrust::raw_pointer_cast(m_previousVel.data()),
		thrust::raw_pointer_cast(m_predictVel.data()));
}