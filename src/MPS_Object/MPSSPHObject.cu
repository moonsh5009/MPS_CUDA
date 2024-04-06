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
	m_factor.clear();
}

void mps::SPHObject::Resize(const size_t size)
{
	mps::ParticleObject::Resize(size);

	m_density.resize(size);
	m_pressure.resize(size);
	m_factor.resize(size);
}

std::shared_ptr<mps::ObjectResource> mps::SPHObject::GenerateDeviceResource()
{
	auto pSuperParam = std::static_pointer_cast<mps::ParticleResource>(ParticleObject::GenerateDeviceResource());
	if (!pSuperParam) return {};

	return std::make_shared<SPHResource>(
		pSuperParam,
		thrust::raw_pointer_cast(m_density.data()),
		thrust::raw_pointer_cast(m_pressure.data()),
		thrust::raw_pointer_cast(m_factor.data()));
}