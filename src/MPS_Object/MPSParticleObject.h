#pragma once

#include "MPSParticleParam.h"
#include "MPSObject.h"

#include "HeaderPre.h"

namespace mps
{
	class __MY_EXT_CLASS__ ParticleObject : public Object
	{
	public:
		ParticleObject();
		~ParticleObject() = default;
		ParticleObject(const ParticleObject&) = delete;
		ParticleObject(ParticleObject&&) = default;
		ParticleObject& operator=(const ParticleObject&) = delete;
		ParticleObject& operator=(ParticleObject&&) = default;

	public:
		virtual void Clear() override;
		virtual void Resize(const size_t size) override;

	protected:
		virtual std::shared_ptr<ObjectResource> GenerateDeviceResource();

	public:
		mcuda::gl::Buffer<REAL> m_radius;
	};
}

#include "HeaderPost.h"