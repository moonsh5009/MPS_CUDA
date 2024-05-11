#pragma once

#include "MPSParticleParam.h"

namespace mps
{
	struct SPHParam : public ParticleParam
	{
		REAL* pDensity;
		REAL* pPressure;
		REAL* pFactorDFSPH;
		REAL* pFactorST;
		REAL* pFactorSTB;

		REAL* pTempReal;
		REAL3* pTempVec3;
		REAL3x3* pTempMat3;
		REAL3* pPreviousVel;
		REAL3* pPredictVel;
	};

	struct SPHResource : public ParticleResource
	{
	public:
		SPHResource() = delete;
		SPHResource(std::shared_ptr<ParticleResource> pSuper,
			REAL* density, REAL* pressure, REAL* factorDFSPH, REAL* factorST, REAL* factorSTB, REAL* tempReal, REAL3* tempVec3, REAL3x3* tempMat3, REAL3* previousVel, REAL3* predictVel) :
			ParticleResource{ std::move(*pSuper) }, m_density{ density }, m_pressure{ pressure }, m_factorDFSPH{ factorDFSPH }, m_factorST{ factorST }, m_factorSTB{ factorSTB },
			m_tempReal{ tempReal }, m_tempVec3{ tempVec3 }, m_tempMat3{ tempMat3 }, m_previousVel { previousVel }, m_predictVel{ predictVel }
		{}
		~SPHResource() = default;
		SPHResource(const SPHResource&) = delete;
		SPHResource(SPHResource&&) = default;
		SPHResource& operator=(const SPHResource&) = delete;
		SPHResource& operator=(SPHResource&&) = default;

	public:
		std::weak_ptr<SPHParam> GetSPHParam() const
		{
			return std::static_pointer_cast<SPHParam>(m_pParam);
		}

		virtual void SetParam()
		{
			if (!m_pParam)
				m_pParam = std::make_shared<SPHParam>();

			auto pParam = std::static_pointer_cast<SPHParam>(m_pParam);
			pParam->pDensity = m_density;
			pParam->pPressure = m_pressure;
			pParam->pFactorDFSPH = m_factorDFSPH;
			pParam->pFactorST = m_factorST;
			pParam->pFactorSTB = m_factorSTB;
			pParam->pTempReal = m_tempReal;
			pParam->pTempVec3 = m_tempVec3;
			pParam->pTempMat3 = m_tempMat3;
			pParam->pPreviousVel = m_previousVel;
			pParam->pPredictVel = m_predictVel;
			ParticleResource::SetParam();
		}

	private:
		REAL* m_density;
		REAL* m_pressure;
		REAL* m_factorDFSPH;
		REAL* m_factorST;
		REAL* m_factorSTB;
		
		REAL* m_tempReal;
		REAL3* m_tempVec3;
		REAL3x3* m_tempMat3;
		REAL3* m_previousVel;
		REAL3* m_predictVel;
	};
}