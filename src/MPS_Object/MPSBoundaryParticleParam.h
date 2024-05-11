#pragma once

#include "MPSParticleParam.h"

namespace mps
{
	struct BoundaryParticleParam : public ParticleParam
	{
		uint32_t* pFaceID;
		REAL2* pBCC;
		REAL* pVolume;

		REAL3* pPreviousVel;
		REAL3* pPredictVel;
	};

	struct BoundaryParticleResource : public ParticleResource
	{
	public:
		BoundaryParticleResource() = delete;
		BoundaryParticleResource(std::shared_ptr<ParticleResource> pSuper,
			uint32_t* faceID, REAL2* bcc, REAL* volume, REAL3* previousVel, REAL3* predictVel) :
			ParticleResource{ std::move(*pSuper) }, m_faceID{ faceID }, m_bcc{ bcc }, m_volume{ volume },
			m_pPreviousVel{ previousVel }, m_pPredictVel{ predictVel }
		{}
		~BoundaryParticleResource() = default;
		BoundaryParticleResource(const BoundaryParticleResource&) = delete;
		BoundaryParticleResource(BoundaryParticleResource&&) = default;
		BoundaryParticleResource& operator=(const BoundaryParticleResource&) = delete;
		BoundaryParticleResource& operator=(BoundaryParticleResource&&) = default;

	public:
		std::weak_ptr<BoundaryParticleParam> GetBoundaryParticleParam() const
		{
			return std::static_pointer_cast<BoundaryParticleParam>(m_pParam);
		}

		virtual void SetParam()
		{
			if (!m_pParam)
				m_pParam = std::make_shared<BoundaryParticleParam>();

			auto pParam = std::static_pointer_cast<BoundaryParticleParam>(m_pParam);
			pParam->pFaceID = m_faceID;
			pParam->pBCC = m_bcc;
			pParam->pVolume = m_volume;
			pParam->pPreviousVel = m_pPreviousVel;
			pParam->pPredictVel = m_pPredictVel;
			ParticleResource::SetParam();
		}

	private:
		uint32_t* m_faceID;
		REAL2* m_bcc;
		REAL* m_volume;

		REAL3* m_pPreviousVel;
		REAL3* m_pPredictVel;
	};
}