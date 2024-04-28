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
	m_factorA.clear();
	m_tempVec3.clear();
	m_previousVel.clear();
	m_predictVel.clear();
}

void mps::SPHObject::Resize(const size_t size)
{
	mps::ParticleObject::Resize(size);

	m_density.resize(size);
	m_pressure.resize(size);
	m_factorA.resize(size);
	m_tempVec3.resize(size);
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
		thrust::raw_pointer_cast(m_factorA.data()),
		thrust::raw_pointer_cast(m_tempVec3.data()),
		thrust::raw_pointer_cast(m_previousVel.data()),
		thrust::raw_pointer_cast(m_predictVel.data()));
}