#pragma once

#include "MPSObjectParam.h"

namespace mps
{
	struct ParticleParam : public ObjectParam
	{
		REAL* pRadius;
	};

	struct ParticleResource : public ObjectResource
	{
	public:
		ParticleResource() = delete;
		ParticleResource(std::shared_ptr<ObjectResource> pSuper, mcuda::gl::DeviceResource<REAL>&& radius) :
			ObjectResource{ std::move(*pSuper) }, m_radius{ std::move(radius) }
		{
		}
		~ParticleResource() = default;
		ParticleResource(const ParticleResource&) = delete;
		ParticleResource(ParticleResource&&) = default;
		ParticleResource& operator=(const ParticleResource&) = delete;
		ParticleResource& operator=(ParticleResource&&) = default;

	public:
		std::weak_ptr<ParticleParam> GetParticleParam() const
		{
			return std::static_pointer_cast<ParticleParam>(m_pParam);
		}

		virtual void SetParam()
		{
			if (!m_pParam)
				m_pParam = std::make_shared<ParticleParam>();

			auto pParam = std::static_pointer_cast<ParticleParam>(m_pParam);
			pParam->pRadius = m_radius.GetData();
			ObjectResource::SetParam();
		}

	private:
		mcuda::gl::DeviceResource<REAL> m_radius;
	};
}