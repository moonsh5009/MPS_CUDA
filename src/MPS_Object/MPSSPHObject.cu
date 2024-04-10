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

	m_smallDensity.clear();
	m_smallPressure.clear();
	m_surfaceTensor.clear();
	m_colorField.clear();
}

void mps::SPHObject::Resize(const size_t size)
{
	mps::ParticleObject::Resize(size);

	m_density.resize(size);
	m_pressure.resize(size);
	m_factorA.resize(size);

	m_smallDensity.resize(size);
	m_smallPressure.resize(size);
	m_surfaceTensor.resize(size);
	m_colorField.resize(size);
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
		thrust::raw_pointer_cast(m_smallDensity.data()),
		thrust::raw_pointer_cast(m_smallPressure.data()),
		thrust::raw_pointer_cast(m_surfaceTensor.data()),
		thrust::raw_pointer_cast(m_colorField.data()));
}