#pragma once

#include "MPSParticleParam.h"

namespace mps
{
	class SPHParam : public ParticleParam
	{
	public:
		MCUDA_HOST_DEVICE_FUNC SPHParam() : ParticleParam{},
			m_pDensity{ nullptr }, m_pPressure{ nullptr }, m_pFactorA{ nullptr }, m_pTempVec3{ nullptr }, m_pPreviousVel{ nullptr }, m_pPredictVel{ nullptr }
		{}
		MCUDA_HOST_DEVICE_FUNC ~SPHParam() {};

	public:
		MCUDA_DEVICE_FUNC REAL& Density(uint32_t idx) { return m_pDensity[idx]; }
		MCUDA_DEVICE_FUNC REAL& Pressure(uint32_t idx) { return m_pPressure[idx]; }
		MCUDA_DEVICE_FUNC REAL& FactorA(uint32_t idx) { return m_pFactorA[idx]; }

		MCUDA_DEVICE_FUNC REAL3& TempVec3(uint32_t idx) { return m_pTempVec3[idx]; }
		MCUDA_DEVICE_FUNC REAL3& PreviousVel(uint32_t idx) { return m_pPreviousVel[idx]; }
		MCUDA_DEVICE_FUNC REAL3& PredictVel(uint32_t idx) { return m_pPredictVel[idx]; }

		MCUDA_DEVICE_FUNC REAL Density(uint32_t idx) const { return m_pDensity[idx]; }
		MCUDA_DEVICE_FUNC REAL Pressure(uint32_t idx) const { return m_pPressure[idx]; }
		MCUDA_DEVICE_FUNC REAL FactorA(uint32_t idx) const { return m_pFactorA[idx]; }
		MCUDA_DEVICE_FUNC REAL Volume(uint32_t idx) const
		{
			if (const auto d = Density(idx); d > 1.0e-10)
				return Mass(idx) / d;
			return 0.0;
		}

		MCUDA_DEVICE_FUNC REAL3 TempVec3(uint32_t idx) const { return m_pTempVec3[idx]; }
		MCUDA_DEVICE_FUNC REAL3 PreviousVel(uint32_t idx) const { return m_pPreviousVel[idx]; }
		MCUDA_DEVICE_FUNC REAL3 PredictVel(uint32_t idx) const { return m_pPredictVel[idx]; }

	public:
		MCUDA_HOST_FUNC REAL* GetDensityArray() const { return m_pDensity; }
		MCUDA_HOST_FUNC REAL* GetPressureArray() const { return m_pPressure; }
		MCUDA_HOST_FUNC REAL* GetFactorAArray() const { return m_pFactorA; }

		MCUDA_HOST_FUNC REAL3* GetTempVec3Array() const { return m_pTempVec3; }
		MCUDA_HOST_FUNC REAL3* GetPreviousVelArray() const { return m_pPreviousVel; }
		MCUDA_HOST_FUNC REAL3* GetPredictVelArray() const { return m_pPredictVel; }

		MCUDA_HOST_FUNC void SetDensityArray(REAL* pDensity) { m_pDensity = pDensity; }
		MCUDA_HOST_FUNC void SetPressureArray(REAL* pPressure) { m_pPressure = pPressure; }
		MCUDA_HOST_FUNC void SetFactorAArray(REAL* pFactorA) { m_pFactorA = pFactorA; }

		MCUDA_HOST_FUNC void SetTempVec3Array(REAL3* pTempVec3) { m_pTempVec3 = pTempVec3; }
		MCUDA_HOST_FUNC void SetPreviousVelArray(REAL3* pPreviousVel) { m_pPreviousVel = pPreviousVel; }
		MCUDA_HOST_FUNC void SetPredictVelArray(REAL3* pPredictVel) { m_pPredictVel = pPredictVel; }

	private:
		REAL* MCUDA_RESTRICT m_pDensity;
		REAL* MCUDA_RESTRICT m_pPressure;
		REAL* MCUDA_RESTRICT m_pFactorA;

		REAL3* MCUDA_RESTRICT m_pTempVec3;
		REAL3* MCUDA_RESTRICT m_pPreviousVel;
		REAL3* MCUDA_RESTRICT m_pPredictVel;
	};

	struct SPHResource : public ParticleResource
	{
	public:
		SPHResource() = delete;
		SPHResource(std::shared_ptr<ParticleResource> pSuper,
			REAL* density, REAL* pressure, REAL* factorA, REAL3* tempVec3, REAL3* previousVel, REAL3* predictVel) :
			ParticleResource{ std::move(*pSuper) }, m_density{ density }, m_pressure{ pressure }, m_factorA{ factorA }, 
			m_tempVec3{ tempVec3 }, m_pPreviousVel { previousVel }, m_pPredictVel{ predictVel }
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
			pParam->SetDensityArray(m_density);
			pParam->SetPressureArray(m_pressure);
			pParam->SetFactorAArray(m_factorA);

			pParam->SetTempVec3Array(m_tempVec3);
			pParam->SetPreviousVelArray(m_pPreviousVel);
			pParam->SetPredictVelArray(m_pPredictVel);
			ParticleResource::SetParam();
		}

	private:
		REAL* m_density;
		REAL* m_pressure;
		REAL* m_factorA;

		REAL3* m_tempVec3;
		REAL3* m_pPreviousVel;
		REAL3* m_pPredictVel;
	};
}