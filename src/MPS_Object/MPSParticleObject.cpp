#include "stdafx.h"
#include "MPSParticleObject.h"

mps::ParticleObject::ParticleObject() : Object{}
{
}

void mps::ParticleObject::Clear()
{
	Object::Clear();

	m_radius.Clear();
}

void mps::ParticleObject::Resize(const size_t size)
{
	Object::Resize(size);

	m_radius.Resize(size);
}

std::shared_ptr<mps::ObjectResource> mps::ParticleObject::GenerateDeviceResource()
{
	auto pSuperParam = Object::GenerateDeviceResource();
	if (!pSuperParam) return {};

	auto radiusRes = m_radius.GetDeviceResource();
	if (!radiusRes) return {};

	return std::make_shared<ParticleResource>(pSuperParam, std::move(radiusRes.value()));
}
