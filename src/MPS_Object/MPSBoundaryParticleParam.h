#pragma once

#include "MPSParticleParam.h"

namespace mps
{
	class BoundaryParticleParam : public ParticleParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC BoundaryParticleParam() : ParticleParam{},
			m_pFaceID{ nullptr }, m_pBCC{ nullptr }, m_pVolume{ nullptr }, m_pPreviousVel{ nullptr }, m_pPredictVel{ nullptr }
		{}
		MCUDA_HOST_DEVICE_FUNC ~BoundaryParticleParam() {}

	public:
		MCUDA_DEVICE_FUNC uint32_t& FaceID(uint32_t idx) { return m_pFaceID[idx]; }
		MCUDA_DEVICE_FUNC REAL2& BCC(uint32_t idx) { return m_pBCC[idx]; }
		MCUDA_DEVICE_FUNC REAL& Volume(uint32_t idx) { return m_pVolume[idx]; }

		MCUDA_DEVICE_FUNC REAL3& PreviousVel(uint32_t idx) { return m_pPreviousVel[idx]; }
		MCUDA_DEVICE_FUNC REAL3& PredictVel(uint32_t idx) { return m_pPredictVel[idx]; }

		MCUDA_DEVICE_FUNC uint32_t FaceID(uint32_t idx) const { return m_pFaceID[idx]; }
		MCUDA_DEVICE_FUNC REAL2 BCC(uint32_t idx) const { return m_pBCC[idx]; }
		MCUDA_DEVICE_FUNC REAL Volume(uint32_t idx) const { return m_pVolume[idx]; }

		MCUDA_DEVICE_FUNC REAL3 PreviousVel(uint32_t idx) const { return m_pPreviousVel[idx]; }
		MCUDA_DEVICE_FUNC REAL3 PredictVel(uint32_t idx) const { return m_pPredictVel[idx]; }

	public:
		MCUDA_HOST_FUNC uint32_t* GetFaceIDArray() const { return m_pFaceID; }
		MCUDA_HOST_FUNC REAL2* GetBCCArray() const { return m_pBCC; }
		MCUDA_HOST_FUNC REAL* GetVolumeArray() const { return m_pVolume; }

		MCUDA_HOST_FUNC REAL3* GetPreviousVelArray() const { return m_pPreviousVel; }
		MCUDA_HOST_FUNC REAL3* GetPredictVelArray() const { return m_pPredictVel; }

		MCUDA_HOST_FUNC void SetFaceIDArray(uint32_t* pFaceID) { m_pFaceID = pFaceID; }
		MCUDA_HOST_FUNC void SetBCCArray(REAL2* pBCC) { m_pBCC = pBCC; }
		MCUDA_HOST_FUNC void SetVolumeArray(REAL* pVolume) { m_pVolume = pVolume; }

		MCUDA_HOST_FUNC void SetPreviousVelArray(REAL3* pPreviousVel) { m_pPreviousVel = pPreviousVel; }
		MCUDA_HOST_FUNC void SetPredictVelArray(REAL3* pPredictVel) { m_pPredictVel = pPredictVel; }

	private:
		uint32_t* MCUDA_RESTRICT m_pFaceID;
		REAL2* MCUDA_RESTRICT m_pBCC;
		REAL* MCUDA_RESTRICT m_pVolume;

		REAL3* MCUDA_RESTRICT m_pPreviousVel;
		REAL3* MCUDA_RESTRICT m_pPredictVel;
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
			pParam->SetFaceIDArray(m_faceID);
			pParam->SetBCCArray(m_bcc);
			pParam->SetVolumeArray(m_volume);

			pParam->SetPreviousVelArray(m_pPreviousVel);
			pParam->SetPredictVelArray(m_pPredictVel);
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