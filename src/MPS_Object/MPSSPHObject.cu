#include "stdafx.h"
#include "MPSSPHObject.h"

mps::SPHObject::SPHObject() : mps::ParticleObject{}
{
}

void mps::SPHObject::Clear()
{
	mps::ParticleObject::Clear();

	m_mass.clear();
	m_density.clear();
	m_pressure.clear();
	m_factor.clear();
}

void mps::SPHObject::Resize(const size_t size)
{
	mps::ParticleObject::Resize(size);

	m_mass.resize(size);
	m_density.resize(size);
	m_pressure.resize(size);
	m_factor.resize(size);
}

std::shared_ptr<mps::ObjectResource> mps::SPHObject::GetObjectResource()
{
	/*auto pSuperParam = std::dynamic_pointer_cast<ParticleResource>(ParticleObject::GetObjectResource());
	if (!pSuperParam) return {};

	return std::make_shared<SPHResource>(
		pSuperParam,
		m_mass.begin(),
		m_density.begin(),
		m_pressure.begin(),
		m_factor.begin());*/

	auto pSuperParam = std::dynamic_pointer_cast<ParticleResource>(ParticleObject::GetObjectResource());
	if (!pSuperParam) return {};

	return std::make_shared<SPHResource>(
		pSuperParam,
		nullptr,
		nullptr,
		nullptr,
		nullptr);
}