#include "stdafx.h"
#include "MPSParticleObject.h"

mps::ParticleObject::ParticleObject() : mps::Object{}
{
}

void mps::ParticleObject::Clear()
{
	mps::Object::Clear();

	m_pos.Clear();
	m_radius.Clear();
	m_color.Clear();
}

void mps::ParticleObject::Resize(const size_t size)
{
	mps::Object::Resize(size);

	m_pos.Resize(size);
	m_radius.Resize(size);
	m_color.Resize(size);
}

std::shared_ptr<mps::ObjectResource> mps::ParticleObject::GetObjectResource()
{
	if (auto posRes = m_pos.GetDeviceResource())
	{
		if (auto radiusRes = m_radius.GetDeviceResource())
		{
			if (auto colorRes = m_color.GetDeviceResource())
			{
				return std::make_shared<ParticleResource>(
					Object::GetObjectResource(),
					std::move(posRes.value()),
					std::move(radiusRes.value()),
					std::move(colorRes.value()));
			}
		}
	}

	assert(false);
	return {};
}
