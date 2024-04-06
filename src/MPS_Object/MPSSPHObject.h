#pragma once

#include "MPSSPHParam.h"
#include "MPSParticleObject.h"

#include "HeaderPre.h"

namespace mps
{
	class __MY_EXT_CLASS__ SPHObject : public ParticleObject
	{
	public:
		SPHObject();
		~SPHObject() = default;
		SPHObject(const SPHObject&) = delete;
		SPHObject(SPHObject&&) = default;
		SPHObject& operator=(const SPHObject&) = delete;
		SPHObject& operator=(SPHObject&&) = default;

	public:
		virtual void Clear() override;
		virtual void Resize(const size_t size) override;

	protected:
		virtual std::shared_ptr<ObjectResource> GenerateDeviceResource();

	public:
		thrust::device_vector<REAL> m_density;
		thrust::device_vector<REAL> m_pressure;
		thrust::device_vector<REAL> m_factor;
	};
}

#include "HeaderPost.h"