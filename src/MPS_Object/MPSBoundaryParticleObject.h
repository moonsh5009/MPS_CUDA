#pragma once

#include "MPSBoundaryParticleParam.h"
#include "MPSParticleObject.h"

#include "HeaderPre.h"

namespace mps
{
	class __MY_EXT_CLASS__ BoundaryParticleObject : public ParticleObject
	{
	public:
		BoundaryParticleObject();
		~BoundaryParticleObject() = default;
		BoundaryParticleObject(const BoundaryParticleObject&) = delete;
		BoundaryParticleObject(BoundaryParticleObject&&) = default;
		BoundaryParticleObject& operator=(const BoundaryParticleObject&) = delete;
		BoundaryParticleObject& operator=(BoundaryParticleObject&&) = default;

	public:
		virtual void Clear() override;
		virtual void Resize(const size_t size) override;

	protected:
		virtual std::shared_ptr<ObjectResource> GenerateDeviceResource();

	public:
		thrust::device_vector<uint32_t> m_faceID;
		thrust::device_vector<REAL2> m_bcc;
		thrust::device_vector<REAL> m_volume;

		thrust::device_vector<REAL3> m_previousVel;
		thrust::device_vector<REAL3> m_predictVel;
	};
}

#include "HeaderPost.h"