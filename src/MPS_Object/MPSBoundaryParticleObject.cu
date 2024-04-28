#include "stdafx.h"
#include "MPSBoundaryParticleObject.h"

mps::BoundaryParticleObject::BoundaryParticleObject() : mps::ParticleObject{}
{
}

void mps::BoundaryParticleObject::Clear()
{
	mps::ParticleObject::Clear();

	m_faceID.clear();
	m_bcc.clear();
	m_volume.clear();
	m_previousVel.clear();
	m_predictVel.clear();
}

void mps::BoundaryParticleObject::Resize(const size_t size)
{
	mps::ParticleObject::Resize(size);

	m_faceID.resize(size);
	m_bcc.resize(size);
	m_volume.resize(size);
	m_previousVel.resize(size);
	m_predictVel.resize(size);
}

std::shared_ptr<mps::ObjectResource> mps::BoundaryParticleObject::GenerateDeviceResource()
{
	auto pSuperParam = std::static_pointer_cast<mps::ParticleResource>(ParticleObject::GenerateDeviceResource());
	if (!pSuperParam) return {};

	return std::make_shared<BoundaryParticleResource>(
		pSuperParam,
		thrust::raw_pointer_cast(m_faceID.data()),
		thrust::raw_pointer_cast(m_bcc.data()),
		thrust::raw_pointer_cast(m_volume.data()),
		thrust::raw_pointer_cast(m_previousVel.data()),
		thrust::raw_pointer_cast(m_predictVel.data()));
}