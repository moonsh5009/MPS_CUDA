#pragma once

#include "MPSObjectParam.h"

namespace mps
{
	class ParticleParam : public ObjectParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC ParticleParam() : ObjectParam{}, m_pRadius{ nullptr } {};
		MCUDA_HOST_DEVICE_FUNC ~ParticleParam() {}

	public:
		MCUDA_DEVICE_FUNC REAL& Radius(uint32_t idx) { return m_pRadius[idx]; }
		MCUDA_DEVICE_FUNC const REAL& Radius(uint32_t idx) const { return m_pRadius[idx]; }

	public:
		MCUDA_HOST_FUNC REAL* GetRadiusArray() const { return m_pRadius; }
		MCUDA_HOST_FUNC void SetRadiusArray(REAL* pRadius) { m_pRadius = pRadius; }

	private:
		REAL* m_pRadius;
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
			pParam->SetRadiusArray(m_radius.ptr);
			ObjectResource::SetParam();
		}

	private:
		mcuda::gl::DeviceResource<REAL> m_radius;
	};
}